"""V18: v17 LGBM + MLP ensemble blend.

Strategy:
- Build features exactly as v17 (CF target + veering/icing + K=80 selection)
- Train v17 LGBM specialists as before
- Additionally train an MLP ensemble (5 seeds) on the same feature set + CF target
- Search for optimal blend weight on Fold-5, apply to valid predictions

Rationale:
- MLP adds non-tree function diversity to the ensemble
- Tabular DL usually loses to LGBM alone, but a small blend can help
- CF target + standardized features = clean DL setup

Usage:
    python -m src.training.train_v18_mlp_blend
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v18.0_mlp_blend.csv"

SEEDS_SPEC = [42, 123, 456, 789, 2026]
MLP_SEEDS = [42, 123, 456, 789, 2026]
K = 80
TURBINE_RATED_MW = 3.465
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


# ---------- Feature pipeline helpers (same as v17) ----------

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


# ---------- MLP model ----------

class PowerMLP(nn.Module):
    """Simple MLP for tabular CF regression.

    3 hidden layers with BatchNorm + Dropout. Output sigmoid to bound CF in [0, 1].
    """

    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_mlp_single(X_tr, y_tr, X_va, y_va, seed, max_epochs=200, patience=25, batch_size=512, lr=1e-3):
    """Train one MLP seed; return best predictions on X_va."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PowerMLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.L1Loss()  # MAE matches the metric

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)

    n = len(X_tr_t)
    best_val = float("inf")
    best_preds = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(X_va_t)
            va_loss = loss_fn(va_pred, y_va_t).item()

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_preds = va_pred.cpu().numpy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_preds, best_val


def train_mlp_ensemble(X_tr, y_tr, X_va, y_va, seeds):
    """Train an ensemble of MLPs; return averaged predictions."""
    preds = []
    for s in seeds:
        p, v = train_mlp_single(X_tr, y_tr, X_va, y_va, s)
        preds.append(p)
        print(f"    MLP seed {s}: val CF loss {v:.5f}")
    return np.mean(preds, axis=0)


def train_mlp_full(X, y, n_epochs, X_test, seeds):
    """Train MLPs on full data (fixed epochs), predict on X_test."""
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        model = PowerMLP(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        loss_fn = nn.L1Loss()
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        n = len(X_t)
        batch_size = 512
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
    return np.mean(preds, axis=0)


# ---------- LGBM helpers ----------

def train_lgbm_specialist_valid(X_tr, y_tr, X_va, y_va, feat_cols, seeds, weights):
    preds = []
    best_iters = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        dt = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
        dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        best_iters.append(b.best_iteration)
    return np.mean(preds, axis=0), best_iters


def train_lgbm_specialist_full(X, y, feat_cols, seeds, n_rounds, X_test, weights):
    preds = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        dt = lgb.Dataset(X, label=y, weight=weights, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dt, num_boost_round=n_rounds)
        preds.append(b.predict(X_test))
    return np.mean(preds, axis=0)


def main():
    set_global_seed(42)
    print("=" * 60)
    print("V18: LGBM specialists + MLP blend")
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

    # --- Fold-5 setup ---
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
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(y_tr_mw, active_tr)
    y_va_cf = to_cf(y_va_mw, active_va)

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)

    # Probe for top-K.
    print(f"\nProbe for top-{K} features...")
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # === LGBM specialists ===
    print("\n=== Fold-5: LGBM specialists ===")
    regime_preds_cf = {}
    regime_iters = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        print(f"Training specialist {name}...")
        preds_cf, iters = train_lgbm_specialist_valid(X_tr, y_tr_cf, X_va, y_va_cf, top_k, SEEDS_SPEC, weights)
        regime_preds_cf[name] = preds_cf
        regime_iters[name] = iters

    lgbm_avg3_cf = np.mean(list(regime_preds_cf.values()), axis=0)
    lgbm_avg3_mw = np.clip(from_cf(lgbm_avg3_cf, active_va), 0, CAPACITY_MW)
    lgbm_nmae = normalized_mae(y_va_mw, lgbm_avg3_mw)
    print(f"  LGBM avg3 Fold-5: {lgbm_nmae:.4f}%")

    # === MLP ensemble ===
    print("\n=== Fold-5: MLP ensemble ===")
    # Standardize features for MLP.
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_std = (X_tr - mu) / sigma
    X_va_std = (X_va - mu) / sigma
    # Fill any NaN (from edge cases).
    X_tr_std = np.nan_to_num(X_tr_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_va_std = np.nan_to_num(X_va_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    mlp_preds_cf = train_mlp_ensemble(X_tr_std, y_tr_cf, X_va_std, y_va_cf, MLP_SEEDS)
    mlp_mw = np.clip(from_cf(mlp_preds_cf, active_va), 0, CAPACITY_MW)
    mlp_nmae = normalized_mae(y_va_mw, mlp_mw)
    print(f"  MLP ensemble Fold-5: {mlp_nmae:.4f}%")

    # === Blend search ===
    print("\n=== Blend search ===")
    best_w = 0.0
    best_nmae = lgbm_nmae
    for w in np.arange(0.0, 0.55, 0.05):
        blend_cf = (1 - w) * lgbm_avg3_cf + w * mlp_preds_cf
        blend_mw = np.clip(from_cf(blend_cf, active_va), 0, CAPACITY_MW)
        nmae = normalized_mae(y_va_mw, blend_mw)
        marker = " <-- best" if nmae < best_nmae else ""
        print(f"  w_mlp={w:.2f}: {nmae:.4f}%{marker}")
        if nmae < best_nmae:
            best_nmae = nmae
            best_w = w
    print(f"\nBest blend: w_mlp={best_w:.2f} -> Fold-5 {best_nmae:.4f}%")
    print(f"Baseline LGBM avg3:      {lgbm_nmae:.4f}%")
    print(f"MLP alone:                {mlp_nmae:.4f}%")
    print(f"Delta vs LGBM:            {best_nmae - lgbm_nmae:+.4f} pp")

    # === Full-fit for submission ===
    print("\n" + "=" * 60)
    print("Full-fit for submission")
    print("=" * 60)

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

    # LGBM specialists full fit.
    specialist_valid_cf = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_full >= lo) & (ws_full < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        n_rounds_spec = max(int(np.median(regime_iters[name]) * 1.2), 2000)
        print(f"  LGBM {name}: {len(SEEDS_SPEC)} seeds, {n_rounds_spec} rounds")
        p = train_lgbm_specialist_full(X_full, y_full_cf, top_k, SEEDS_SPEC, n_rounds_spec, X_valid, weights)
        specialist_valid_cf[name] = p
    lgbm_valid_cf = np.mean(list(specialist_valid_cf.values()), axis=0)

    # MLP full fit.
    print(f"  MLP: {len(MLP_SEEDS)} seeds on full training data")
    mu_full = X_full.mean(axis=0)
    sigma_full = X_full.std(axis=0)
    sigma_full[sigma_full < 1e-6] = 1.0
    X_full_std = ((X_full - mu_full) / sigma_full).astype(np.float32)
    X_valid_std = ((X_valid - mu_full) / sigma_full).astype(np.float32)
    X_full_std = np.nan_to_num(X_full_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_valid_std = np.nan_to_num(X_valid_std, nan=0.0, posinf=0.0, neginf=0.0)
    # Use a reasonable fixed epoch count based on Fold-5 training.
    mlp_valid_cf = train_mlp_full(X_full_std, y_full_cf, 80, X_valid_std, MLP_SEEDS)

    # Final blend.
    final_cf = (1 - best_w) * lgbm_valid_cf + best_w * mlp_valid_cf
    preds_valid_mw = np.clip(from_cf(final_cf, active_valid), 0, CAPACITY_MW)

    # Restore original row order.
    order = df_vp["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_valid_mw)
    po[order] = preds_valid_mw
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))
    print(f"\n  Submission saved: {SUBMISSION_PATH}")
    print(f"  Blend weight: w_mlp={best_w:.2f}")
    print(f"  Mean prediction: {preds_valid_mw.mean():.2f} MW")
    print(f"  Expected Fold-5 nMAE: {best_nmae:.4f}%")
    print("Done.")


if __name__ == "__main__":
    main()
