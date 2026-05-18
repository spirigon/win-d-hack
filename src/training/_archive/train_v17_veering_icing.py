"""V17: v16 (CF target + regime specialists) + enhanced veering & icing features.

New features added to pipeline.py:
- Signed veering (4 pairs: 10-80, 80-120, 120-180, 10-120) — stability signal
- Icing severity (continuous), icing × wind, freezing rain, snow accum, rime proxy

Evaluation: compare Fold-5 with and without new features to measure impact.

Usage:
    python -m src.training.train_v17_veering_icing
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
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TOTAL_TURBINES
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
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v17.0_veering_icing.csv"

SEEDS_BASE = [42, 123, 456, 789, 2026, 3141, 1618, 2718, 7777, 12345]
SEEDS_SPEC = [42, 123, 456, 789, 2026]
K = 80  # Bumped from 70 to include veering/icing features at positions 75-83
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
    """Convert MW to capacity factor."""
    p_available = active_turbines * TURBINE_RATED_MW
    return y_mw / np.maximum(p_available, 1e-3)


def from_cf(cf, active_turbines):
    """Convert capacity factor back to MW."""
    p_available = active_turbines * TURBINE_RATED_MW
    return cf * p_available


def train_seed_ensemble_valid(X_tr, y_tr, X_va, y_va, feat_cols, seeds, weights=None):
    """Train seed ensemble, return predictions on X_va + best_iters."""
    preds = []
    best_iters = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        if weights is not None:
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=weights, feature_name=feat_cols, free_raw_data=False)
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        p = b.predict(X_va, num_iteration=b.best_iteration)
        preds.append(p)
        best_iters.append(b.best_iteration)
    return np.mean(preds, axis=0), best_iters


def train_seed_ensemble_full(X, y, feat_cols, seeds, n_rounds, X_test, weights=None):
    """Train on full data (no early stopping), predict on X_test."""
    preds = []
    for s in seeds:
        cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
        if weights is not None:
            dtrain = lgb.Dataset(X, label=y, weight=weights, feature_name=feat_cols, free_raw_data=False)
        else:
            dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)
        p = b.predict(X_test)
        preds.append(p)
    return np.mean(preds, axis=0)


def main():
    set_global_seed(42)
    print("=" * 60)
    print("V17: CF target + regime specialists + veering & icing features")
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

    # --- Fold-5 evaluation ---
    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fold_val = df_train.iloc[val_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]

    # Fit power curves and wake on fold-train.
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_val, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    # Feature columns (exclude bookkeeping).
    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]

    # --- CF target transform ---
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(y_tr_mw, active_tr)
    y_va_cf = to_cf(y_va_mw, active_va)

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)

    # --- Probe for top-K features ---
    print(f"\nProbe run for top-{K} features from {len(feat_cols_all)} total...")
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])

    # Print top-20 and highlight new features.
    new_features = {
        "veer_signed_10m_80m", "veer_signed_80m_120m", "veer_signed_120m_180m",
        "veer_signed_10m_120m", "icing_severity", "icing_x_wind",
        "freezing_rain_risk", "snow_accum_risk", "rime_ice_proxy",
    }
    print("\nTop-20 features by gain:")
    for i, (name, gain) in enumerate(feat_imp[:20]):
        marker = " *** NEW ***" if name in new_features else ""
        print(f"  {i+1:2d}. {name:40s} {gain:12.0f}{marker}")

    # Show where new features rank.
    print("\nNew feature rankings:")
    for i, (name, gain) in enumerate(feat_imp):
        if name in new_features:
            print(f"  #{i+1:3d}  {name:40s} {gain:12.0f}")

    top_k = [n for n, _ in feat_imp[:K]]
    new_in_topk = [n for n in top_k if n in new_features]
    print(f"\nNew features in top-{K}: {len(new_in_topk)} — {new_in_topk}")

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()
    ws_va = df_va["wind_speed_120m"].to_numpy()

    # === Fold-5 evaluation (CF target) ===
    print("\n" + "=" * 60)
    print("Fold-5 evaluation (CF target, regime specialists)")
    print("=" * 60)

    # Base model (10 seeds).
    print("\nTraining base (10 seeds)...")
    preds_base_cf, iters_base = train_seed_ensemble_valid(X_tr, y_tr_cf, X_va, y_va_cf, top_k, SEEDS_BASE)
    preds_base_mw = np.clip(from_cf(preds_base_cf, active_va), 0, CAPACITY_MW)
    nmae_base = normalized_mae(y_va_mw, preds_base_mw)
    print(f"  Base (CF): {nmae_base:.4f}%")

    # 3 regime specialists.
    regime_preds_cf = {}
    regime_iters = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        print(f"Training specialist {name}...")
        preds_cf, iters = train_seed_ensemble_valid(X_tr, y_tr_cf, X_va, y_va_cf, top_k, SEEDS_SPEC, weights)
        regime_preds_cf[name] = preds_cf
        regime_iters[name] = iters
        preds_mw = np.clip(from_cf(preds_cf, active_va), 0, CAPACITY_MW)
        print(f"  {name}: {normalized_mae(y_va_mw, preds_mw):.4f}%")

    # Blend strategies.
    avg3_cf = np.mean(list(regime_preds_cf.values()), axis=0)
    avg3_mw = np.clip(from_cf(avg3_cf, active_va), 0, CAPACITY_MW)

    blend_50_cf = 0.5 * preds_base_cf + 0.5 * avg3_cf
    blend_50_mw = np.clip(from_cf(blend_50_cf, active_va), 0, CAPACITY_MW)

    blend_60_cf = 0.6 * preds_base_cf + 0.4 * avg3_cf
    blend_60_mw = np.clip(from_cf(blend_60_cf, active_va), 0, CAPACITY_MW)

    print("\n=== Fold-5 blend comparison ===")
    print(f"  Base (10 seeds):           {nmae_base:.4f}%")
    print(f"  Avg 3 specialists:         {normalized_mae(y_va_mw, avg3_mw):.4f}%")
    print(f"  Base + avg3 (50/50):       {normalized_mae(y_va_mw, blend_50_mw):.4f}%")
    print(f"  Base + avg3 (60/40):       {normalized_mae(y_va_mw, blend_60_mw):.4f}%")

    # Pick best blend for submission.
    blends = {
        "avg3": (avg3_cf, normalized_mae(y_va_mw, avg3_mw)),
        "50/50": (blend_50_cf, normalized_mae(y_va_mw, blend_50_mw)),
        "60/40": (blend_60_cf, normalized_mae(y_va_mw, blend_60_mw)),
    }
    best_name = min(blends, key=lambda k: blends[k][1])
    print(f"\n  Best blend: {best_name} ({blends[best_name][1]:.4f}%)")

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

    # Base.
    n_rounds_base = max(int(np.median(iters_base) * 1.2), 2000)
    print(f"  Base: {len(SEEDS_BASE)} seeds, {n_rounds_base} rounds")
    preds_valid_base_cf = train_seed_ensemble_full(X_full, y_full_cf, top_k, SEEDS_BASE, n_rounds_base, X_valid)

    # Specialists.
    specialist_valid_cf = {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask_in = (ws_full >= lo) & (ws_full < hi)
        weights = np.where(mask_in, 2.0, 0.3).astype(np.float32)
        n_rounds_spec = max(int(np.median(regime_iters[name]) * 1.2), 2000)
        print(f"  {name}: {len(SEEDS_SPEC)} seeds, {n_rounds_spec} rounds")
        p = train_seed_ensemble_full(X_full, y_full_cf, top_k, SEEDS_SPEC, n_rounds_spec, X_valid, weights)
        specialist_valid_cf[name] = p

    # Apply best blend.
    avg3_valid_cf = np.mean(list(specialist_valid_cf.values()), axis=0)
    if best_name == "avg3":
        final_cf = avg3_valid_cf
    elif best_name == "50/50":
        final_cf = 0.5 * preds_valid_base_cf + 0.5 * avg3_valid_cf
    else:
        final_cf = 0.6 * preds_valid_base_cf + 0.4 * avg3_valid_cf

    # Convert CF -> MW.
    preds_valid_mw = np.clip(from_cf(final_cf, active_valid), 0, CAPACITY_MW)

    # Restore original row order.
    order = df_vp["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_valid_mw)
    po[order] = preds_valid_mw
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))
    print(f"\n  Submission saved: {SUBMISSION_PATH}")
    print(f"  Mean prediction: {preds_valid_mw.mean():.2f} MW")
    print(f"  Std prediction:  {preds_valid_mw.std():.2f} MW")
    print("\nDone.")


if __name__ == "__main__":
    main()
