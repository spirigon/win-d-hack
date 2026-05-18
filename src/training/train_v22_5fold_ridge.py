"""V22: 5-fold CV-bagged LGBM with optional Ridge stacker.

Motivation from LB diagnostics:
- v19.2 (full-fit): LB 7.678
- v20.1 (3-fold CV-bag): LB 7.630 (-0.048 pp)
- v21 (+ AV reweight): identical to v20 (AV has no effect)

Hypothesis: more CV-bagging diversity -> better averaging -> lower LB.
Also test a Ridge meta-stacker on OOF predictions (more principled than
grid simplex).

Strategy:
- Use all 5 folds (F1..F5) with early stopping + seed ensemble
- Produce OOF predictions for all training rows (where each fold contributes
  its validation set)
- Fit a non-negative Ridge meta-stacker on OOF -> predictions blend
- CV-bag test predictions (average 5 fold-models' test predictions)

Usage:
    python -m src.training.train_v22_5fold_ridge
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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

SEEDS_SPEC = [42, 123, 456]  # 3 seeds per regime specialist
K = 80
FOLD_IDS = [0, 1, 2, 3, 4]  # All 5 folds
TURBINE_RATED_MW = 3.465

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
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


def train_spec_fold(X_tr, y_tr, X_va, y_va, X_test, feat_cols, seeds, weights):
    """Train specialist (one regime) with weights. Return (val, test) avg preds."""
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


def main():
    set_global_seed(42)
    print("=" * 60)
    print("V22: 5-fold CV-bag LGBM")
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

    folds = default_folds()

    # Feature probe from Fold-5.
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

    # --- 5-fold loop ---
    test_preds_per_fold = []
    oof_rows = []

    for fold_idx in FOLD_IDS:
        fold = folds[fold_idx]
        print(f"\n{'=' * 60}\nFold {fold_idx+1}\n{'=' * 60}")
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

        regime_val_cf = {}
        regime_test_cf = {}
        for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
            mask_in = (ws_tr >= lo) & (ws_tr < hi)
            weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
            val_p, test_p = train_spec_fold(X_tr, y_tr_cf, X_va, y_va_cf, X_test, top_k, SEEDS_SPEC, weights)
            regime_val_cf[name] = val_p
            regime_test_cf[name] = test_p
            print(f"  {name}: val={normalized_mae(y_va_mw, np.clip(from_cf(val_p, active_va), 0, CAPACITY_MW)):.4f}%")

        lgbm_val_cf = np.mean(list(regime_val_cf.values()), axis=0)
        lgbm_test_cf = np.mean(list(regime_test_cf.values()), axis=0)
        lgbm_val_mw = np.clip(from_cf(lgbm_val_cf, active_va), 0, CAPACITY_MW)
        fold_nmae = normalized_mae(y_va_mw, lgbm_val_mw)
        print(f"  Fold {fold_idx+1} avg3: {fold_nmae:.4f}%")

        oof_rows.append({
            "y_mw": y_va_mw, "active": active_va, "cf": lgbm_val_cf,
            "low": regime_val_cf["low_0_7"],
            "mid": regime_val_cf["mid_4_12"],
            "high": regime_val_cf["high_8_25"],
        })
        test_preds_per_fold.append({
            "low": regime_test_cf["low_0_7"],
            "mid": regime_test_cf["mid_4_12"],
            "high": regime_test_cf["high_8_25"],
            "avg": lgbm_test_cf,
        })

    # --- OOF aggregation ---
    print("\n" + "=" * 60)
    print("OOF summary (all 5 folds)")
    print("=" * 60)

    y_all = np.concatenate([r["y_mw"] for r in oof_rows])
    active_all = np.concatenate([r["active"] for r in oof_rows])
    cf_all = np.concatenate([r["cf"] for r in oof_rows])
    low_all = np.concatenate([r["low"] for r in oof_rows])
    mid_all = np.concatenate([r["mid"] for r in oof_rows])
    high_all = np.concatenate([r["high"] for r in oof_rows])

    mw_all = np.clip(from_cf(cf_all, active_all), 0, CAPACITY_MW)
    print(f"  avg3 OOF: {normalized_mae(y_all, mw_all):.4f}%")

    # Per-fold detail.
    print("\n  Per-fold:")
    for i, r in enumerate(oof_rows):
        fmw = np.clip(from_cf(r["cf"], r["active"]), 0, CAPACITY_MW)
        print(f"    Fold {FOLD_IDS[i]+1}: {normalized_mae(r['y_mw'], fmw):.4f}%")

    # --- Ridge stacker on OOF regime preds ---
    print("\n--- Ridge meta-stacker on OOF (low/mid/high -> final) ---")
    from sklearn.linear_model import Ridge
    stack_X = np.column_stack([low_all, mid_all, high_all])
    # Target: actual CF for y_all
    y_cf_all = to_cf(y_all, active_all)

    # Non-negative constrained LinearRegression via positive Ridge.
    from scipy.optimize import nnls
    # Include an intercept term as a constant column.
    A = np.column_stack([stack_X, np.ones(len(stack_X))])
    coefs, _ = nnls(A, y_cf_all)
    print(f"  NNLS coefs: low={coefs[0]:.4f}  mid={coefs[1]:.4f}  high={coefs[2]:.4f}  bias={coefs[3]:.4f}")

    # Also try constrained to sum=1 (no bias).
    # Use Ridge with non-negative constraints approximated via clipping.
    ridge = Ridge(alpha=0.1, positive=True)
    ridge.fit(stack_X, y_cf_all)
    print(f"  Positive Ridge coefs: low={ridge.coef_[0]:.4f}  mid={ridge.coef_[1]:.4f}  high={ridge.coef_[2]:.4f}  intercept={ridge.intercept_:.5f}")

    # Evaluate stacker.
    ridge_oof_cf = ridge.predict(stack_X)
    ridge_oof_mw = np.clip(from_cf(ridge_oof_cf, active_all), 0, CAPACITY_MW)
    print(f"  Ridge OOF:    {normalized_mae(y_all, ridge_oof_mw):.4f}%")
    nnls_oof_cf = A @ coefs
    nnls_oof_mw = np.clip(from_cf(nnls_oof_cf, active_all), 0, CAPACITY_MW)
    print(f"  NNLS OOF:     {normalized_mae(y_all, nnls_oof_mw):.4f}%")

    # Also try constrained-sum-to-1 simplex search.
    best = (None, float("inf"))
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6:
                continue
            cf_ = w1 * low_all + w2 * mid_all + w3 * high_all
            mw_ = np.clip(from_cf(cf_, active_all), 0, CAPACITY_MW)
            nmae = normalized_mae(y_all, mw_)
            if nmae < best[1]:
                best = ((w1, w2, w3), nmae)
    print(f"  Simplex best: w={best[0]} -> {best[1]:.4f}%")

    # --- Produce submissions ---
    print("\n--- Test predictions ---")
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)

    # Average across 5 folds for each regime specialist.
    low_test = np.mean([tp["low"] for tp in test_preds_per_fold], axis=0)
    mid_test = np.mean([tp["mid"] for tp in test_preds_per_fold], axis=0)
    high_test = np.mean([tp["high"] for tp in test_preds_per_fold], axis=0)
    avg_test = np.mean([tp["avg"] for tp in test_preds_per_fold], axis=0)

    # Default avg3 submission.
    avg_mw = np.clip(from_cf(avg_test, active_valid), 0, CAPACITY_MW)
    po = np.empty_like(avg_mw)
    po[order] = avg_mw
    write_submission(po, _ROOT / "submissions" / "archive" / "v22.0_5fold_avg3.csv", expected_rows=len(df_valid))
    print(f"  Saved v22.0_5fold_avg3.csv  mean={avg_mw.mean():.2f} MW")

    # Ridge stacked.
    stack_test = np.column_stack([low_test, mid_test, high_test])
    ridge_test_cf = ridge.predict(stack_test)
    ridge_mw = np.clip(from_cf(ridge_test_cf, active_valid), 0, CAPACITY_MW)
    po2 = np.empty_like(ridge_mw)
    po2[order] = ridge_mw
    write_submission(po2, _ROOT / "submissions" / "archive" / "v22.1_5fold_ridge.csv", expected_rows=len(df_valid))
    print(f"  Saved v22.1_5fold_ridge.csv  mean={ridge_mw.mean():.2f} MW")

    # Simplex-optimal.
    w1, w2, w3 = best[0]
    simplex_test_cf = w1 * low_test + w2 * mid_test + w3 * high_test
    simplex_mw = np.clip(from_cf(simplex_test_cf, active_valid), 0, CAPACITY_MW)
    po3 = np.empty_like(simplex_mw)
    po3[order] = simplex_mw
    write_submission(po3, _ROOT / "submissions" / "archive" / "v22.2_5fold_simplex.csv", expected_rows=len(df_valid))
    print(f"  Saved v22.2_5fold_simplex.csv  mean={simplex_mw.mean():.2f} MW")

    print("\nDone.")


if __name__ == "__main__":
    main()
