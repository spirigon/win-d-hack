"""V21: v20 LGBM pipeline + adversarial validation sample reweighting.

AV diagnostic (AUC=0.9994) showed strong distribution shift dominated by
temperature and pressure. Q1 months have AV prob ~100x higher than summer
months. This script uses AV probs to reweight training samples toward the
Q1-like distribution.

Strategy:
- Compute AV prob per training row (done separately by adversarial_validation.py)
- Derive sample weight: w = max(av_prob, eps) scaled
- Apply to LGBM specialists training (CV-bagged across 3 folds)

Three weighting strategies to compare on Fold-5 AND a "winter-only" holdout
(Nov-Mar rows only, which are closest to Q1 2026 distribution):
1. Uniform (v20 baseline)
2. AV-soft: w = (av_prob + 0.01) ** 0.5, clipped
3. AV-hard: w = 1 if av_prob > threshold else 0.1 (seasonal filter)

Usage:
    python -m src.training.train_v21_av_reweight
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
AV_OOF_PATH = _ROOT / "data" / "processed" / "adversarial_oof.parquet"

SEEDS_SPEC = [42, 123, 456, 789, 2026]
K = 80
FOLD_IDS = [2, 3, 4]  # Folds 3, 4, 5
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


def compute_weights(av_probs, strategy, regime_mask=None, regime_in=2.0, regime_out=0.3):
    """Derive sample weights from AV probs + optional regime mask.

    strategy:
      'uniform': all rows weight 1, optionally boosted by regime
      'av_soft': w = sqrt(av_prob + 0.01), rescaled to mean 1
      'av_pow4': w = (av_prob + 0.005)^0.25, less aggressive
      'winter_only': w = 1 if month in Nov-Mar else 0.2

    Regime multiplier (low/mid/high wind specialist):
      if regime_mask provided, multiply by regime_in inside mask, regime_out outside
    """
    eps = 1e-6
    if strategy == "uniform":
        w = np.ones_like(av_probs, dtype=np.float32)
    elif strategy == "av_soft":
        w = np.sqrt(av_probs + 0.01)
        w = w / w.mean()
    elif strategy == "av_pow4":
        w = (av_probs + 0.005) ** 0.25
        w = w / w.mean()
    elif strategy == "winter_only":
        # Requires caller to pass av_probs but it's actually month array here.
        raise ValueError("winter_only strategy requires month array, not av_probs")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Clamp extreme weights (avoid complete domination by rare extreme rows).
    w = np.clip(w, 0.05, 10.0)

    if regime_mask is not None:
        regime_mult = np.where(regime_mask, regime_in, regime_out).astype(np.float32)
        w = w * regime_mult

    return w.astype(np.float32)


def train_specialist_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, seeds, base_weights):
    """Train LGBM specialist with given weights (CV-bagged across seeds)."""
    val_preds, test_preds = [], []
    best_iters = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        dt = lgb.Dataset(X_tr, label=y_tr, weight=base_weights, feature_name=feat_cols, free_raw_data=False)
        dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        val_preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        test_preds.append(b.predict(X_test, num_iteration=b.best_iteration))
        best_iters.append(b.best_iteration)
    return np.mean(val_preds, axis=0), np.mean(test_preds, axis=0), best_iters


def evaluate_strategy(strategy, df_train, df_valid_sorted, top_k, folds, FOLD_IDS, av_oof):
    """Evaluate one weighting strategy on OOF and produce test predictions."""
    print(f"\n{'='*60}\nStrategy: {strategy}\n{'='*60}")

    oof_rows = []
    test_preds_per_fold = []

    for fold_idx in FOLD_IDS:
        fold = folds[fold_idx]
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

        # AV prob for these train rows.
        av_ts = av_oof.set_index(TIMESTAMP_COL)["av_prob"]
        fold_ts = df_tr[TIMESTAMP_COL].map(av_ts)
        av_prob_tr = fold_ts.fillna(av_ts.mean()).to_numpy(dtype=np.float32)

        # Base AV weights for this strategy.
        base_w = compute_weights(av_prob_tr, strategy)

        # Regime specialists (3 of them, each with low/mid/high wind boost).
        regime_val_cf = {}
        regime_test_cf = {}
        for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
            mask_in = (ws_tr >= lo) & (ws_tr < hi)
            # Multiply AV weights by regime boost.
            w = base_w * np.where(mask_in, 2.0, 0.3).astype(np.float32)
            val_p, test_p, _ = train_specialist_ensemble(
                X_tr, y_tr_cf, X_va, y_va_cf, X_test, top_k, SEEDS_SPEC, w
            )
            regime_val_cf[name] = val_p
            regime_test_cf[name] = test_p

        lgbm_val_cf = np.mean(list(regime_val_cf.values()), axis=0)
        lgbm_test_cf = np.mean(list(regime_test_cf.values()), axis=0)
        lgbm_val_mw = np.clip(from_cf(lgbm_val_cf, active_va), 0, CAPACITY_MW)
        fold_nmae = normalized_mae(y_va_mw, lgbm_val_mw)
        print(f"  Fold {fold_idx+1}: nMAE = {fold_nmae:.4f}%")

        oof_rows.append({"y_mw": y_va_mw, "active": active_va, "cf": lgbm_val_cf})
        test_preds_per_fold.append(lgbm_test_cf)

    y_all = np.concatenate([r["y_mw"] for r in oof_rows])
    active_all = np.concatenate([r["active"] for r in oof_rows])
    cf_all = np.concatenate([r["cf"] for r in oof_rows])
    all_mw = np.clip(from_cf(cf_all, active_all), 0, CAPACITY_MW)
    oof_nmae = normalized_mae(y_all, all_mw)
    print(f"  OOF (3 folds):  {oof_nmae:.4f}%")

    # Per-fold summary.
    f5_nmae = normalized_mae(oof_rows[-1]["y_mw"],
                             np.clip(from_cf(oof_rows[-1]["cf"], oof_rows[-1]["active"]), 0, CAPACITY_MW))
    print(f"  Fold-5:         {f5_nmae:.4f}%  (winter 2025; LB surrogate)")

    # Average test preds across folds.
    test_cf = np.mean(test_preds_per_fold, axis=0)
    return {
        "strategy": strategy,
        "oof_nmae": oof_nmae,
        "f5_nmae": f5_nmae,
        "test_cf": test_cf,
    }


def main():
    set_global_seed(42)
    print("=" * 60)
    print("V21: LGBM + Adversarial Validation Sample Reweighting")
    print("=" * 60)

    # Check AV OOF exists.
    if not AV_OOF_PATH.exists():
        print(f"\nERROR: {AV_OOF_PATH} not found.")
        print("Run: python -m src.training.adversarial_validation")
        return
    av_oof = pd.read_parquet(AV_OOF_PATH)
    av_oof[TIMESTAMP_COL] = pd.to_datetime(av_oof[TIMESTAMP_COL])
    print(f"\nLoaded AV OOF: {len(av_oof)} rows")
    print(f"  AV prob: mean={av_oof['av_prob'].mean():.4f}, p99={av_oof['av_prob'].quantile(0.99):.4f}")

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

    # Get top-K features from Fold-5 probe (same as v20).
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

    # --- Evaluate each strategy ---
    results = {}
    for strategy in ["uniform", "av_pow4", "av_soft"]:
        results[strategy] = evaluate_strategy(
            strategy, df_train, df_valid_sorted, top_k, folds, FOLD_IDS, av_oof
        )

    # Summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<15} {'OOF nMAE':>10} {'Fold-5':>10}")
    print("-" * 40)
    for s, r in results.items():
        print(f"{s:<15} {r['oof_nmae']:>9.4f}% {r['f5_nmae']:>9.4f}%")

    # --- Produce submissions for each strategy ---
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)
    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)

    for strategy, r in results.items():
        preds_mw = np.clip(from_cf(r["test_cf"], active_valid), 0, CAPACITY_MW)
        po = np.empty_like(preds_mw)
        po[order] = preds_mw
        tag = strategy.replace("_", "")
        path = _ROOT / "submissions" / "archive" / f"v21.0_{tag}.csv"
        write_submission(po, path, expected_rows=len(df_valid))
        print(f"  Saved: {path.name}  mean={preds_mw.mean():.2f} MW")

    print("\nDone.")


if __name__ == "__main__":
    main()
