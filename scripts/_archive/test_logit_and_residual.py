"""Test two ideas on Fold-5:
B. Logit target transform: model logit(CF) instead of CF
C. 2-stage residual: train second LGBM on first's residuals

Both are quick to test and could close the 0.185 pp gap to first place.

Usage:
    python scripts/test_logit_and_residual.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

_ROOT = Path(__file__).resolve().parents[1]
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
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"

K = 80
TURBINE_RATED_MW = 3.465

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)

# Stage-2 config: deeper, slower, more regularized.
CONFIG_S2 = LGBMConfig(
    num_leaves=31, min_data_in_leaf=30, learning_rate=0.005,
    feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=5,
    lambda_l1=1.0, lambda_l2=1.0,
    num_boost_round=5000, early_stopping_rounds=200, log_period=0,
)


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


def train_specialists(X_tr, y_tr, X_va, feat_cols, ws_tr, seeds, config, weights_base=None):
    """Train 3 regime specialists, return val predictions."""
    regime_preds = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask, 2.0, 0.3).astype(np.float32)
        if weights_base is not None:
            regime_w = regime_w * weights_base
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**config.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=regime_w, feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=np.zeros(len(X_va)), feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=config.num_boost_round,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)])
            preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds.append(np.mean(preds, axis=0))
    return np.mean(regime_preds, axis=0)


def main():
    set_global_seed(42)
    print("=" * 70)
    print("Testing: B (logit target) + C (2-stage residual)")
    print("=" * 70)

    df_train = load_train(TRAIN_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    df_train["_split"] = "train"
    combined = df_train.copy()
    combined = build_features(combined, sort_by_time=True)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)
    df_train = combined.reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

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

    # Feature probe.
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dt = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dv = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    seeds = [42, 123, 456, 789, 2026]

    # ============================================================
    # BASELINE: CF target (v20 config)
    # ============================================================
    print("\n--- A. Baseline (CF target, MAE) ---")
    regime_preds = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights, feature_name=top_k, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds.append(np.mean(preds, axis=0))
    base_cf = np.mean(regime_preds, axis=0)
    base_mw = np.clip(from_cf(base_cf, active_va), 0, CAPACITY_MW)
    base_nmae = normalized_mae(y_va_mw, base_mw)
    print(f"  Baseline CF: {base_nmae:.4f}%")

    # ============================================================
    # B. LOGIT TARGET
    # ============================================================
    print("\n--- B. Logit target transform ---")
    # Transform CF to logit space.
    eps = 0.001
    y_tr_cf_clipped = np.clip(y_tr_cf, eps, 1.0 - eps)
    y_tr_logit = logit(y_tr_cf_clipped).astype(np.float32)

    regime_preds_logit = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr_logit, weight=weights, feature_name=top_k, free_raw_data=False)
            dv_logit = lgb.Dataset(X_va, label=logit(np.clip(y_va_cf, eps, 1-eps)), feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                          valid_sets=[dv_logit], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            pred_logit = b.predict(X_va, num_iteration=b.best_iteration)
            # Inverse transform: logit -> CF.
            pred_cf = expit(pred_logit)
            preds.append(pred_cf)
        regime_preds_logit.append(np.mean(preds, axis=0))
    logit_cf = np.mean(regime_preds_logit, axis=0)
    logit_mw = np.clip(from_cf(logit_cf, active_va), 0, CAPACITY_MW)
    logit_nmae = normalized_mae(y_va_mw, logit_mw)
    print(f"  Logit target: {logit_nmae:.4f}% (Δ={logit_nmae - base_nmae:+.4f})")

    # Also try log1p transform.
    print("\n--- B2. Log1p target transform ---")
    y_tr_log = np.log1p(y_tr_cf * 100).astype(np.float32)  # scale CF to 0-100 range first
    regime_preds_log = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr_log, weight=weights, feature_name=top_k, free_raw_data=False)
            dv_log = lgb.Dataset(X_va, label=np.log1p(y_va_cf * 100), feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                          valid_sets=[dv_log], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            pred_log = b.predict(X_va, num_iteration=b.best_iteration)
            pred_cf = (np.expm1(pred_log)) / 100.0
            preds.append(pred_cf)
        regime_preds_log.append(np.mean(preds, axis=0))
    log_cf = np.mean(regime_preds_log, axis=0)
    log_mw = np.clip(from_cf(log_cf, active_va), 0, CAPACITY_MW)
    log_nmae = normalized_mae(y_va_mw, log_mw)
    print(f"  Log1p target: {log_nmae:.4f}% (Δ={log_nmae - base_nmae:+.4f})")

    # ============================================================
    # C. 2-STAGE RESIDUAL
    # ============================================================
    print("\n--- C. 2-stage residual correction ---")
    # Stage 1: standard CF prediction (already computed as base_cf).
    # Stage 2: predict residual = actual_cf - stage1_cf.
    # Need OOF predictions from stage 1 for training stage 2.

    # Quick OOF: use the probe model's predictions on training data (leaky but fast diagnostic).
    # Better: use leave-one-out or a separate fold split.
    # For speed: split training into 2 halves, predict each half with the other.
    n_tr = len(X_tr)
    half = n_tr // 2
    oof_stage1 = np.zeros(n_tr, dtype=np.float32)

    # First half predicts second half and vice versa.
    for s in seeds[:1]:  # Single seed for speed.
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        # Train on first half, predict second.
        dt1 = lgb.Dataset(X_tr[:half], label=y_tr_cf[:half], feature_name=top_k, free_raw_data=False)
        dv1 = lgb.Dataset(X_tr[half:], label=y_tr_cf[half:], feature_name=top_k, free_raw_data=False)
        b1 = lgb.train(cfg.to_params(), dt1, num_boost_round=3000,
                       valid_sets=[dv1], valid_names=["val"],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_stage1[half:] = b1.predict(X_tr[half:], num_iteration=b1.best_iteration)
        # Train on second half, predict first.
        dt2 = lgb.Dataset(X_tr[half:], label=y_tr_cf[half:], feature_name=top_k, free_raw_data=False)
        dv2 = lgb.Dataset(X_tr[:half], label=y_tr_cf[:half], feature_name=top_k, free_raw_data=False)
        b2 = lgb.train(cfg.to_params(), dt2, num_boost_round=3000,
                       valid_sets=[dv2], valid_names=["val"],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_stage1[:half] = b2.predict(X_tr[:half], num_iteration=b2.best_iteration)

    # Residual target.
    residual_tr = y_tr_cf - oof_stage1

    # Stage 2 features: original features + stage1 prediction.
    X_tr_s2 = np.column_stack([X_tr, oof_stage1])
    X_va_s2 = np.column_stack([X_va, base_cf])  # Use stage1 val predictions.
    feat_cols_s2 = top_k + ["stage1_pred"]

    # Train stage 2 on residuals.
    s2_preds = []
    for s in seeds:
        cfg2 = LGBMConfig(**{**CONFIG_S2.__dict__, "seed": s})
        dt = lgb.Dataset(X_tr_s2, label=residual_tr, feature_name=feat_cols_s2, free_raw_data=False)
        dv = lgb.Dataset(X_va_s2, label=(y_va_cf - base_cf), feature_name=feat_cols_s2, free_raw_data=False)
        b = lgb.train(cfg2.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        s2_preds.append(b.predict(X_va_s2, num_iteration=b.best_iteration))
    residual_pred = np.mean(s2_preds, axis=0)

    # Final = stage1 + stage2 correction.
    final_cf = base_cf + residual_pred
    final_mw = np.clip(from_cf(final_cf, active_va), 0, CAPACITY_MW)
    residual_nmae = normalized_mae(y_va_mw, final_mw)
    print(f"  2-stage residual: {residual_nmae:.4f}% (Δ={residual_nmae - base_nmae:+.4f})")

    # Also try blending stage1 + (stage1 + stage2).
    for alpha in [0.3, 0.5, 0.7]:
        blend_cf = (1 - alpha) * base_cf + alpha * final_cf
        blend_mw = np.clip(from_cf(blend_cf, active_va), 0, CAPACITY_MW)
        blend_nmae = normalized_mae(y_va_mw, blend_mw)
        print(f"    Blend alpha={alpha}: {blend_nmae:.4f}% (Δ={blend_nmae - base_nmae:+.4f})")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY (Fold-5)")
    print("=" * 70)
    print(f"  Baseline (CF, MAE):    {base_nmae:.4f}%")
    print(f"  Logit target:          {logit_nmae:.4f}%  ({logit_nmae - base_nmae:+.4f})")
    print(f"  Log1p target:          {log_nmae:.4f}%  ({log_nmae - base_nmae:+.4f})")
    print(f"  2-stage residual:      {residual_nmae:.4f}%  ({residual_nmae - base_nmae:+.4f})")
    print("\nDone.")


if __name__ == "__main__":
    main()
