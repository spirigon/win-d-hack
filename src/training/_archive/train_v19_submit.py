"""V19 submission: LGBM + MLP + FTT blend (skip ResNet, it underperformed).

Based on Fold-5 results:
- FTT alone: 7.5195% (beats LGBM!)
- MLP+FTT pair: 7.4687% (best DL combo)
- LGBM 30% + DL_avg 70%: 7.5007%

Strategy: use MLP+FTT average as the DL component, blend with LGBM.
Optimal weight from Fold-5: w_dl ~ 0.65-0.70.

Usage:
    python -m src.training.train_v19_submit
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v19.0_deep_blend.csv"

SEEDS_SPEC = [42, 123, 456, 789, 2026]
DL_SEEDS = [42, 123, 456, 789, 2026]
K = 80
TURBINE_RATED_MW = 3.465
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Blend weight: 35% LGBM + 65% DL (MLP+FTT avg).
# Slightly conservative vs the 0.70 optimum to avoid overfitting Fold-5.
W_DL = 0.65

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


# --- Models ---

class PowerMLP(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.15):
        super().__init__()
        self.feature_embeddings = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        tokens = self.feature_embeddings(x.unsqueeze(-1))
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        h = self.encoder(tokens)
        cls_out = self.norm(h[:, 0, :])
        return torch.sigmoid(self.head(cls_out)).squeeze(-1)


# --- Training ---

def train_dl_full(model_cls, model_kwargs, X, y, n_epochs, X_test, seeds, batch_size=512, lr=1e-3):
    """Train DL on full data (fixed epochs), predict on X_test."""
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        model = model_cls(**model_kwargs).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        loss_fn = nn.L1Loss()
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        n = len(X_t)
        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                pred = model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            preds.append(model(X_test_t).cpu().numpy())
        print(f"    {model_cls.__name__} seed {s} done ({n_epochs} epochs)")
    return np.mean(preds, axis=0)


# --- Feature pipeline ---

def _add_pc(df, pc_sector, pc_global):
    df = df.copy()
    v_eff = df["v_eff"].to_numpy()
    dir_deg = (df["wind_direction_120m"] * 1000.0).to_numpy()
    df["p_curve_sector"] = pc_sector.predict(v_eff, dir_deg)
    df["p_curve_global"] = pc_global.predict(v_eff)
    df["p_curve_rews"] = pc_global.predict(df["rews"].to_numpy())
    df["p_curve_x_active"] = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_global_x_active"] = df["p_curve_global"] * df["active_turbines_ratio"]
    df["p_curve_ratio"] = df["p_curve_sector"] / CAPACITY_MW
    df["p_curve_sector_minus_global"] = df["p_curve_sector"] - df["p_curve_global"]
    return df


def merge_era5(df, era5):
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def add_era5_rolling(df):
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]
    for w in [3, 6, 12, 24]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"] = roll.std().fillna(0)
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)
    df["era5_pressure_diff3"] = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"] = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)
    roll6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)
    dir_sin = df["era5_dir100_sin"]
    dir_cos = df["era5_dir100_cos"]
    df["era5_dir_sin_diff1"] = dir_sin.diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = dir_cos.diff(1).fillna(0)
    return df


def to_cf(y_mw, active_turbines):
    p_available = active_turbines * TURBINE_RATED_MW
    return y_mw / np.maximum(p_available, 1e-3)


def from_cf(cf, active_turbines):
    p_available = active_turbines * TURBINE_RATED_MW
    return cf * p_available


def main():
    set_global_seed(42)
    print("=" * 60)
    print(f"V19 submission: LGBM {1-W_DL:.0%} + DL(MLP+FTT) {W_DL:.0%}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\nPreparing data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Get feature selection from Fold-5 probe.
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(df_va[TARGET_COL].to_numpy(dtype=np.float32), active_va)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)

    print(f"Probe for top-{K} features...")
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    # --- Full-fit ---
    print("\n--- Full training ---")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    wake_full = fit_wake_lookup(df_train_clean, n_sectors=16)

    df_train_full = _add_pc(df_train, pc_sector_full, pc_global_full)
    df_train_full = add_wake_features(df_train_full, wake_full)
    X_full = df_train_full[top_k].to_numpy(dtype=np.float32)
    active_full = df_train_full["active_turbines"].to_numpy(dtype=np.float32)
    y_full_mw = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    y_full_cf = to_cf(y_full_mw, active_full)
    ws_full = df_train_full["wind_speed_120m"].to_numpy()

    df_vp = _add_pc(df_valid_sorted, pc_sector_full, pc_global_full)
    df_vp = add_wake_features(df_vp, wake_full)
    for c in set(top_k) - set(df_vp.columns):
        df_vp[c] = 0.0
    X_valid = df_vp[top_k].to_numpy(dtype=np.float32)
    active_valid = df_vp["active_turbines"].to_numpy(dtype=np.float32)

    # LGBM specialists.
    print("\nLGBM specialists (full fit):")
    specialist_valid_cf = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_full >= lo) & (ws_full < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in SEEDS_SPEC:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dt = lgb.Dataset(X_full, label=y_full_cf, weight=weights, feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=2200)
            preds.append(b.predict(X_valid))
        specialist_valid_cf[name] = np.mean(preds, axis=0)
        print(f"  {name}: done")
    lgbm_valid_cf = np.mean(list(specialist_valid_cf.values()), axis=0)

    # DL models.
    mu_full = X_full.mean(axis=0)
    sigma_full = X_full.std(axis=0)
    sigma_full[sigma_full < 1e-6] = 1.0
    X_full_std = np.nan_to_num((X_full - mu_full) / sigma_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_valid_std = np.nan_to_num((X_valid - mu_full) / sigma_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    n_features = X_full_std.shape[1]
    n_epochs = 90  # FTT best epochs were 18-86, so 90 covers it

    print(f"\nMLP full ({n_epochs} epochs):")
    mlp_kwargs = {"input_dim": n_features, "hidden": (256, 128, 64), "dropout": 0.2}
    mlp_valid_cf = train_dl_full(PowerMLP, mlp_kwargs, X_full_std, y_full_cf, n_epochs, X_valid_std, DL_SEEDS)

    print(f"\nFTT full ({n_epochs} epochs):")
    ftt_kwargs = {"input_dim": n_features, "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.15}
    ftt_valid_cf = train_dl_full(FTTransformer, ftt_kwargs, X_full_std, y_full_cf, n_epochs, X_valid_std, DL_SEEDS)

    # DL average (MLP + FTT).
    dl_valid_cf = (mlp_valid_cf + ftt_valid_cf) / 2.0

    # Final blend.
    final_cf = (1 - W_DL) * lgbm_valid_cf + W_DL * dl_valid_cf
    preds_valid_mw = np.clip(from_cf(final_cf, active_valid), 0, CAPACITY_MW)

    # Restore original row order.
    order = df_vp["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_valid_mw)
    po[order] = preds_valid_mw
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))
    print(f"\nSubmission saved: {SUBMISSION_PATH}")
    print(f"Blend: LGBM {1-W_DL:.0%} + DL(MLP+FTT) {W_DL:.0%}")
    print(f"Mean prediction: {preds_valid_mw.mean():.2f} MW")
    print(f"Std prediction:  {preds_valid_mw.std():.2f} MW")

    # Also save pure-FTT and pure-LGBM for manual blending.
    ftt_mw = np.clip(from_cf(ftt_valid_cf, active_valid), 0, CAPACITY_MW)
    po_ftt = np.empty_like(ftt_mw)
    po_ftt[order] = ftt_mw
    write_submission(po_ftt, _ROOT / "submissions" / "archive" / "v19.1_ftt_only.csv", expected_rows=len(df_valid))

    lgbm_mw = np.clip(from_cf(lgbm_valid_cf, active_valid), 0, CAPACITY_MW)
    po_lgbm = np.empty_like(lgbm_mw)
    po_lgbm[order] = lgbm_mw
    write_submission(po_lgbm, _ROOT / "submissions" / "archive" / "v19.2_lgbm_only.csv", expected_rows=len(df_valid))

    print("Done.")


if __name__ == "__main__":
    main()
