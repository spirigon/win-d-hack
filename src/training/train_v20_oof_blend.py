"""V20: OOF-based DL blend with proper fold-level regularization.

Key fix from v19: DL models were overfit because full-fit trained for 90 fixed
epochs vs fold early-stop best of 4-40 epochs.

Strategy:
- For each of 3 folds (3, 4, 5):
  - Train LGBM specialists with early stopping; predict OOF + test
  - Train MLP (3 seeds) with early stopping; predict OOF + test
  - Train FTT (3 seeds) with early stopping; predict OOF + test
- Average 3 test predictions per model (CV bagging) -> more robust than full-fit
- Stack 3 fold OOFs -> find blend weight minimizing OOF nMAE across all rows
- Apply optimal weight to test

Also: stronger DL regularization (higher dropout, smaller FTT) to reduce
overfitting to individual fold patterns.

Usage:
    python -m src.training.train_v20_oof_blend
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
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v20.0_oof_blend.csv"

SEEDS_SPEC = [42, 123, 456]  # Reduced from 5 for speed; 3 seeds x 3 folds = 9 total per regime
DL_SEEDS = [42, 123, 456]
K = 80
FOLD_IDS = [2, 3, 4]  # Folds 3, 4, 5 (0-indexed)
TURBINE_RATED_MW = 3.465
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


# ============================================================
# Models (with stronger regularization than v19)
# ============================================================

class PowerMLP(nn.Module):
    """MLP with higher dropout for better generalization."""

    def __init__(self, input_dim, hidden=(192, 96, 48), dropout=0.35):
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
    """Smaller FTT with higher dropout."""

    def __init__(self, input_dim, d_model=48, n_heads=4, n_layers=2, dropout=0.30):
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


def train_dl_fold(model_cls, model_kwargs, X_tr, y_tr, X_va, y_va, X_test, seed,
                  max_epochs=150, patience=25, batch_size=512, lr=1e-3, weight_decay=5e-4):
    """Train DL with early stopping on val, return (val_preds, test_preds, best_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_cls(**model_kwargs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.L1Loss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    n = len(X_tr_t)
    best_val = float("inf")
    best_val_preds = None
    best_test_preds = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(X_va_t)
            va_loss = loss_fn(va_pred, y_va_t).item()

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_val_preds = va_pred.cpu().numpy()
            with torch.no_grad():
                best_test_preds = model(X_test_t).cpu().numpy()
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_preds, best_test_preds, best_val, best_epoch


def train_dl_ensemble_fold(model_cls, model_kwargs, X_tr, y_tr, X_va, y_va, X_test, seeds, name="Model"):
    """Train ensemble across seeds; return (avg_val_preds, avg_test_preds)."""
    val_preds, test_preds = [], []
    for s in seeds:
        vp, tp, vl, ep = train_dl_fold(model_cls, model_kwargs, X_tr, y_tr, X_va, y_va, X_test, s)
        val_preds.append(vp)
        test_preds.append(tp)
        print(f"      {name} seed {s}: val_loss={vl:.5f} best_epoch={ep}")
    return np.mean(val_preds, axis=0), np.mean(test_preds, axis=0)


# ============================================================
# Feature pipeline (identical to v19)
# ============================================================

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


def train_lgbm_spec_fold(X_tr, y_tr, X_va, y_va, X_test, feat_cols, seeds, weights):
    """Train LGBM specialist (one regime) with early stopping; return (val, test) preds."""
    val_preds, test_preds = [], []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        dt = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
        dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        val_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
    return np.mean(val_preds, axis=0), np.mean(test_preds, axis=0)


# ============================================================
# Main
# ============================================================

def build_full_features(df_train, df_valid, era5):
    df_train = df_train.copy()
    df_valid = df_valid.copy()
    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)
    df_train_out = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_out = combined[combined["_split"] == "valid"].reset_index(drop=True)
    return df_train_out, df_valid_out


def main():
    set_global_seed(42)
    print("=" * 70)
    print("V20: OOF-based LGBM + MLP + FTT blend with fold-level early stopping")
    print(f"Device: {DEVICE}")
    print(f"Folds used: {[i+1 for i in FOLD_IDS]}")
    print("=" * 70)

    print("\nPreparing data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    df_train, df_valid_sorted = build_full_features(df_train, df_valid, era5)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    folds = default_folds()

    # --- Feature selection via Fold-5 probe ---
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train_ = df_train.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = _add_pc(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(df_train.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)
    feat_cols_all = [c for c in feature_columns(df_t_) if c not in ("_is_impossible", "_split")]
    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    y_v_cf_ = to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    X_t_all_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_all_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)

    print(f"Probe for top-{K} features...")
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_t_all_, label=y_t_cf_, feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_v_all_, label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt_, num_boost_round=5000,
                      valid_sets=[dv_], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    print(f"  Selected {len(top_k)} features")

    # --- OOF loop across folds ---
    oof_rows = []  # will collect (y_mw, active, lgbm_cf, mlp_cf, ftt_cf) per fold val row
    lgbm_test_preds = []
    mlp_test_preds = []
    ftt_test_preds = []

    for fold_idx in FOLD_IDS:
        fold = folds[fold_idx]
        print(f"\n{'=' * 70}")
        print(f"Fold {fold_idx + 1}: {fold.train_end} | {fold.val_start} -> {fold.val_end}")
        print("=" * 70)

        train_idx, val_idx = split_indices(df_train, fold)
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
        df_te = _add_pc(df_valid_sorted, pc_sector, pc_global)
        df_te = add_wake_features(df_te, wake)
        for c in set(top_k) - set(df_te.columns):
            df_te[c] = 0.0

        active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
        active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
        y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_cf = to_cf(y_va_mw, active_va)

        X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()

        # LGBM specialists (3 regimes x 3 seeds = 9 models; avg by regime -> avg3).
        print(f"\n  LGBM specialists:")
        regime_val_cf = {}
        regime_test_cf = {}
        for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
            mask_in = (ws_tr >= lo) & (ws_tr < hi)
            weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
            val_p, test_p = train_lgbm_spec_fold(X_tr, y_tr_cf, X_va, y_va_cf, X_test,
                                                  top_k, SEEDS_SPEC, weights)
            regime_val_cf[name] = val_p
            regime_test_cf[name] = test_p
            val_mw = np.clip(from_cf(val_p, active_va), 0, CAPACITY_MW)
            print(f"    {name}: val nMAE = {normalized_mae(y_va_mw, val_mw):.4f}%")
        lgbm_val_cf = np.mean(list(regime_val_cf.values()), axis=0)
        lgbm_test_cf = np.mean(list(regime_test_cf.values()), axis=0)
        lgbm_val_mw = np.clip(from_cf(lgbm_val_cf, active_va), 0, CAPACITY_MW)
        print(f"    LGBM avg3: val nMAE = {normalized_mae(y_va_mw, lgbm_val_mw):.4f}%")

        # DL models: standardize features.
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        X_tr_std = np.nan_to_num((X_tr - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_va_std = np.nan_to_num((X_va - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_test_std = np.nan_to_num((X_test - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        n_features = X_tr_std.shape[1]

        print(f"\n  MLP ({len(DL_SEEDS)} seeds):")
        mlp_kwargs = {"input_dim": n_features, "hidden": (192, 96, 48), "dropout": 0.35}
        mlp_val_cf, mlp_test_cf = train_dl_ensemble_fold(
            PowerMLP, mlp_kwargs, X_tr_std, y_tr_cf, X_va_std, y_va_cf, X_test_std, DL_SEEDS, "MLP"
        )
        mlp_val_mw = np.clip(from_cf(mlp_val_cf, active_va), 0, CAPACITY_MW)
        print(f"    MLP avg: val nMAE = {normalized_mae(y_va_mw, mlp_val_mw):.4f}%")

        print(f"\n  FTT ({len(DL_SEEDS)} seeds):")
        ftt_kwargs = {"input_dim": n_features, "d_model": 48, "n_heads": 4, "n_layers": 2, "dropout": 0.30}
        ftt_val_cf, ftt_test_cf = train_dl_ensemble_fold(
            FTTransformer, ftt_kwargs, X_tr_std, y_tr_cf, X_va_std, y_va_cf, X_test_std, DL_SEEDS, "FTT"
        )
        ftt_val_mw = np.clip(from_cf(ftt_val_cf, active_va), 0, CAPACITY_MW)
        print(f"    FTT avg: val nMAE = {normalized_mae(y_va_mw, ftt_val_mw):.4f}%")

        # Store OOF predictions.
        oof_rows.append({
            "y_mw": y_va_mw,
            "active": active_va,
            "lgbm_cf": lgbm_val_cf,
            "mlp_cf": mlp_val_cf,
            "ftt_cf": ftt_val_cf,
        })
        lgbm_test_preds.append(lgbm_test_cf)
        mlp_test_preds.append(mlp_test_cf)
        ftt_test_preds.append(ftt_test_cf)

    # --- OOF blend search ---
    print("\n" + "=" * 70)
    print("OOF blend search")
    print("=" * 70)

    y_all = np.concatenate([r["y_mw"] for r in oof_rows])
    active_all = np.concatenate([r["active"] for r in oof_rows])
    lgbm_all_cf = np.concatenate([r["lgbm_cf"] for r in oof_rows])
    mlp_all_cf = np.concatenate([r["mlp_cf"] for r in oof_rows])
    ftt_all_cf = np.concatenate([r["ftt_cf"] for r in oof_rows])

    lgbm_all_mw = np.clip(from_cf(lgbm_all_cf, active_all), 0, CAPACITY_MW)
    mlp_all_mw = np.clip(from_cf(mlp_all_cf, active_all), 0, CAPACITY_MW)
    ftt_all_mw = np.clip(from_cf(ftt_all_cf, active_all), 0, CAPACITY_MW)

    print(f"\nOOF single-model scores (across {len(FOLD_IDS)} folds):")
    print(f"  LGBM: {normalized_mae(y_all, lgbm_all_mw):.4f}%")
    print(f"  MLP:  {normalized_mae(y_all, mlp_all_mw):.4f}%")
    print(f"  FTT:  {normalized_mae(y_all, ftt_all_mw):.4f}%")

    # Per-fold scores for context.
    print("\nPer-fold nMAE:")
    print(f"  {'fold':<8}{'LGBM':>10}{'MLP':>10}{'FTT':>10}")
    for i, fold_idx in enumerate(FOLD_IDS):
        r = oof_rows[i]
        lgbm_mw = np.clip(from_cf(r["lgbm_cf"], r["active"]), 0, CAPACITY_MW)
        mlp_mw = np.clip(from_cf(r["mlp_cf"], r["active"]), 0, CAPACITY_MW)
        ftt_mw = np.clip(from_cf(r["ftt_cf"], r["active"]), 0, CAPACITY_MW)
        print(f"  F{fold_idx+1:<7}{normalized_mae(r['y_mw'], lgbm_mw):>9.4f}%{normalized_mae(r['y_mw'], mlp_mw):>9.4f}%{normalized_mae(r['y_mw'], ftt_mw):>9.4f}%")

    # Grid search for optimal blend (LGBM, MLP, FTT weights summing to 1).
    print("\nBlend grid search (simplex):")
    best = (None, float("inf"))
    for w_lgbm in np.arange(0.0, 1.01, 0.05):
        for w_mlp in np.arange(0.0, 1.01 - w_lgbm, 0.05):
            w_ftt = 1.0 - w_lgbm - w_mlp
            if w_ftt < -1e-6:
                continue
            blend_cf = w_lgbm * lgbm_all_cf + w_mlp * mlp_all_cf + w_ftt * ftt_all_cf
            blend_mw = np.clip(from_cf(blend_cf, active_all), 0, CAPACITY_MW)
            nmae = normalized_mae(y_all, blend_mw)
            if nmae < best[1]:
                best = ((w_lgbm, w_mlp, w_ftt), nmae)

    w_lgbm_opt, w_mlp_opt, w_ftt_opt = best[0]
    print(f"\n  Best blend: LGBM={w_lgbm_opt:.2f}  MLP={w_mlp_opt:.2f}  FTT={w_ftt_opt:.2f}")
    print(f"  OOF nMAE:   {best[1]:.4f}%")

    # Print some nearby alternatives for sensitivity check.
    print("\n  LGBM-heavy alternatives (for LB robustness):")
    for w_lgbm_alt in [0.60, 0.70, 0.80, 0.90, 1.00]:
        remaining = 1.0 - w_lgbm_alt
        # Split DL 50/50 MLP/FTT.
        w_mlp_alt = remaining / 2
        w_ftt_alt = remaining / 2
        blend_cf = w_lgbm_alt * lgbm_all_cf + w_mlp_alt * mlp_all_cf + w_ftt_alt * ftt_all_cf
        blend_mw = np.clip(from_cf(blend_cf, active_all), 0, CAPACITY_MW)
        nmae = normalized_mae(y_all, blend_mw)
        print(f"    LGBM={w_lgbm_alt:.2f} MLP={w_mlp_alt:.2f} FTT={w_ftt_alt:.2f}: OOF {nmae:.4f}%")

    # --- Apply blend to test ---
    lgbm_test_cf = np.mean(lgbm_test_preds, axis=0)  # avg across folds (CV bag)
    mlp_test_cf = np.mean(mlp_test_preds, axis=0)
    ftt_test_cf = np.mean(ftt_test_preds, axis=0)

    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)

    # Use the OOF-optimal blend.
    final_cf = w_lgbm_opt * lgbm_test_cf + w_mlp_opt * mlp_test_cf + w_ftt_opt * ftt_test_cf
    preds_valid_mw = np.clip(from_cf(final_cf, active_valid), 0, CAPACITY_MW)

    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_valid_mw)
    po[order] = preds_valid_mw
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))
    print(f"\n  Submission (OOF-optimal) saved: {SUBMISSION_PATH}")
    print(f"  Blend weights: LGBM={w_lgbm_opt:.2f} MLP={w_mlp_opt:.2f} FTT={w_ftt_opt:.2f}")
    print(f"  Mean: {preds_valid_mw.mean():.2f} MW  Std: {preds_valid_mw.std():.2f} MW")

    # Also save conservative variants.
    for tag, w_l in [("conservative", 0.70), ("lgbm_heavy", 0.85)]:
        w_rem = 1.0 - w_l
        final_cf_alt = w_l * lgbm_test_cf + (w_rem / 2) * mlp_test_cf + (w_rem / 2) * ftt_test_cf
        preds_alt_mw = np.clip(from_cf(final_cf_alt, active_valid), 0, CAPACITY_MW)
        po_alt = np.empty_like(preds_alt_mw)
        po_alt[order] = preds_alt_mw
        alt_path = _ROOT / "submissions" / "archive" / f"v20.0_{tag}.csv"
        write_submission(po_alt, alt_path, expected_rows=len(df_valid))

    # Save CV-bagged LGBM alone as a reference (no DL).
    lgbm_mw_valid = np.clip(from_cf(lgbm_test_cf, active_valid), 0, CAPACITY_MW)
    po_lgbm = np.empty_like(lgbm_mw_valid)
    po_lgbm[order] = lgbm_mw_valid
    write_submission(po_lgbm, _ROOT / "submissions" / "archive" / "v20.1_lgbm_cv.csv", expected_rows=len(df_valid))

    print("Done.")


if __name__ == "__main__":
    main()
