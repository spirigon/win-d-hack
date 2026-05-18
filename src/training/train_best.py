"""Reproduce the best submission: target-diversity blend (MW + CF).

Best LB: 7.605 (v27.1_mw50_cf50.csv)

Architecture:
- Two LGBM ensembles trained on different targets:
  1. CF target: CF = P / (active_turbines × 3.465)
  2. Raw MW target: P directly
- Each ensemble: 3-fold CV-bag × 3 regime specialists × 3 seeds = 27 models
- Final: 50/50 blend of CF-ensemble and MW-ensemble predictions

Reproduction:
    python -m src.training.train_best

Outputs:
    submissions/final/best_submission.csv

Note: with 3 seeds this takes ~10 min. For exact reproduction of LB 7.605,
use SEEDS = [42, 123, 456, 789, 2026] (5 seeds, ~20 min).
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

# --- Paths ---
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "final" / "best_submission.csv"

# --- Hyperparameters (Optuna-tuned on v14 feature set) ---
LGBM_PARAMS = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)

# --- Constants ---
SEEDS = [42, 123, 456]  # 3 seeds (fast); use [42, 123, 456, 789, 2026] for 5-seed final
K = 80  # Top features to select
FOLD_IDS = [2, 3, 4]  # Folds 3, 4, 5 (most recent, best for Q1 2026)
TURBINE_RATED_MW = 3.465
BLEND_WEIGHT_MW = 0.50  # 50% MW-target + 50% CF-target


# ============================================================
# Feature pipeline
# ============================================================

def add_power_curve_features(df, pc_sector, pc_global):
    """Add isotonic power curve predictions as features."""
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
    """Merge ERA5 reanalysis and compute bias features."""
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    # NWP-vs-ERA5 bias signatures.
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    # ERA5 derived.
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    # Fill NaN from merge.
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def add_era5_rolling(df):
    """Add ERA5 rolling statistics (critical: #1 feature by importance)."""
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
    """Convert MW to capacity factor."""
    p_available = active_turbines * TURBINE_RATED_MW
    return y_mw / np.maximum(p_available, 1e-3)


def from_cf(cf, active_turbines):
    """Convert capacity factor back to MW."""
    p_available = active_turbines * TURBINE_RATED_MW
    return cf * p_available


# ============================================================
# Training
# ============================================================

def train_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, seeds, config):
    """Train 3 regime specialists with early stopping. Return (val_preds, test_preds)."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        val_preds, test_preds = [], []
        for s in seeds:
            cfg = LGBMConfig(**{**config.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=config.num_boost_round,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)])
            val_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
            test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
        regime_val[name] = np.mean(val_preds, axis=0)
        regime_test[name] = np.mean(test_preds, axis=0)
    return np.mean(list(regime_val.values()), axis=0), np.mean(list(regime_test.values()), axis=0)


def run_cv_bag(df_train, df_valid_sorted, top_k, folds, target_mode="cf"):
    """Run 3-fold CV-bag with given target mode. Return test predictions."""
    test_preds_per_fold = []

    for fold_idx in FOLD_IDS:
        fold = folds[fold_idx]
        train_idx, val_idx = split_indices(df_train, fold)
        fold_train = df_train.iloc[train_idx]
        fold_val = df_train.iloc[val_idx]
        fit_data = fold_train[~fold_train["_is_impossible"]]

        # Fit power curves and wake on clean training data.
        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        wake = fit_wake_lookup(fit_data, n_sectors=16)

        # Add power curve + wake features.
        df_tr = add_power_curve_features(fold_train, pc_sector, pc_global)
        df_tr = add_wake_features(df_tr, wake)
        df_va = add_power_curve_features(fold_val, pc_sector, pc_global)
        df_va = add_wake_features(df_va, wake)
        df_te = add_power_curve_features(df_valid_sorted, pc_sector, pc_global)
        df_te = add_wake_features(df_te, wake)
        for c in set(top_k) - set(df_te.columns):
            df_te[c] = 0.0

        # Prepare arrays.
        active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
        active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr = df_tr["wind_speed_120m"].to_numpy()

        # Target depends on mode.
        if target_mode == "cf":
            y_tr = to_cf(y_tr_mw, active_tr)
            y_va = to_cf(y_va_mw, active_va)
        else:  # "mw"
            y_tr = y_tr_mw
            y_va = y_va_mw

        val_pred, test_pred = train_fold_ensemble(
            X_tr, y_tr, X_va, y_va, X_test, top_k, ws_tr, SEEDS, LGBM_PARAMS
        )
        test_preds_per_fold.append(test_pred)

        # Report fold validation.
        if target_mode == "cf":
            val_mw = np.clip(from_cf(val_pred, active_va), 0, CAPACITY_MW)
        else:
            val_mw = np.clip(val_pred, 0, CAPACITY_MW)
        fold_nmae = normalized_mae(y_va_mw, val_mw)
        print(f"    Fold {fold_idx+1} ({target_mode}): {fold_nmae:.4f}%")

    return np.mean(test_preds_per_fold, axis=0)


# ============================================================
# Main
# ============================================================

def main():
    set_global_seed(42)
    print("=" * 60)
    print("Training best submission: target-diversity blend (MW + CF)")
    print("=" * 60)

    # --- Load and prepare data ---
    print("\n[1/4] Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    # --- Build features on combined train+valid (for rolling continuity) ---
    print("[2/4] Building features...")
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

    # --- Feature selection (probe on Fold-5) ---
    print("[3/4] Feature selection...")
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train_ = df_train.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = add_power_curve_features(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = add_power_curve_features(df_train.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)
    feat_cols_all = [c for c in feature_columns(df_t_) if c not in ("_is_impossible", "_split")]
    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    X_t_ = df_t_[feat_cols_all].to_numpy(dtype=np.float32)
    X_v_ = df_v_[feat_cols_all].to_numpy(dtype=np.float32)

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(X_t_, label=y_t_cf_, feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_v_, label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt_, num_boost_round=5000,
                      valid_sets=[dv_], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    print(f"  Selected {len(top_k)} features from {len(feat_cols_all)} total")

    # --- Train both target variants ---
    print("\n[4/4] Training ensembles...")
    print("\n  --- CF target (3-fold CV-bag) ---")
    test_cf = run_cv_bag(df_train, df_valid_sorted, top_k, folds, target_mode="cf")

    print("\n  --- MW target (3-fold CV-bag) ---")
    test_mw_raw = run_cv_bag(df_train, df_valid_sorted, top_k, folds, target_mode="mw")

    # --- Convert and blend ---
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    pred_cf_mw = np.clip(from_cf(test_cf, active_valid), 0, CAPACITY_MW)
    pred_mw_mw = np.clip(test_mw_raw, 0, CAPACITY_MW)

    final_mw = BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw
    final_mw = np.clip(final_mw, 0, CAPACITY_MW)

    # --- Write submission ---
    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid))

    print(f"\n{'=' * 60}")
    print(f"DONE. Submission: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW")
    print(f"  Std:  {final_mw.std():.2f} MW")
    print(f"  Range: [{final_mw.min():.2f}, {final_mw.max():.2f}]")
    print(f"{'=' * 60}")


def train_and_predict_fold5(seed: int = 42) -> float:
    """Run the best pipeline on Fold-5 only and return the nMAE (%).

    This is the hook consumed by ``scripts/measure_fold5_after_hardening.py``
    for the non-regression guardrail (Requirement 9). It reuses the full
    feature pipeline but trains only on the Fold-5 train split and evaluates
    on the Fold-5 validation window (2025-01-01 → 2025-03-31).

    Returns
    -------
    float
        Fold-5 nMAE in percent (e.g. 7.62).
    """
    from src.data.outliers import compute_training_weights
    set_global_seed(seed)

    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = (
        pd.concat([df_train, df_valid], ignore_index=True)
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Explicit weight policy (Requirement 1.5).
    sample_weight_full = compute_training_weights(df_train, downweight_2022=1.0)

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = add_power_curve_features(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = add_power_curve_features(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(y_tr_mw, active_tr)
    y_va_cf = to_cf(y_va_mw, active_va)
    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    sample_weight_fold = sample_weight_full[train_idx].astype(np.float32)

    # Feature selection probe (CF target, seed-fixed).
    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": seed})
    dt_ = lgb.Dataset(X_tr_all, label=y_tr_cf, weight=sample_weight_fold,
                      feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # CF ensemble on Fold-5.
    regime_val_preds = []
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        regime_w = np.where(mask, 2.0, 0.3).astype(np.float32)
        weights = (regime_w * sample_weight_fold).astype(np.float32)
        seed_preds = []
        for s in SEEDS:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights,
                             feature_name=top_k, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=5000,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(250, verbose=False)],
            )
            seed_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_val_preds.append(np.mean(seed_preds, axis=0))

    val_cf = np.mean(regime_val_preds, axis=0)
    val_mw = np.clip(from_cf(val_cf, active_va), 0, CAPACITY_MW)
    return float(normalized_mae(y_va_mw, val_mw))


if __name__ == "__main__":
    main()
