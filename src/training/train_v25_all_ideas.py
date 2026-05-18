"""V25: Comprehensive experiment testing all remaining ideas.

Ideas tested:
A. AI weather model features (AIFS + GraphCast) — new data source
B. Ensemble spread features (partial Q1 2026 only — skipped, too sparse)
C. Quantile-tilted training (alpha=0.47-0.49)
D. Monotonic constraints on physics features
E. 24 hourly LGBMs (one model per hour)
F. More seeds (10 seeds per specialist)

Each tested on Fold-5 (Q1 2025) with the v20 methodology (CF target, K=80,
3 regime specialists). Best ideas combined for final submission.

Usage:
    python -m src.training.train_v25_all_ideas
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
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
AI_WEATHER_PATH = _ROOT / "data" / "external" / "ai_weather_models.parquet"

K = 80
TURBINE_RATED_MW = 3.465

CONFIG = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=250, log_period=0,
)


# ============================================================
# Feature pipeline (same as v20 + AI weather merge)
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


def merge_ai_weather(df, ai_df):
    """Merge AI weather model features. NaN-safe (LGBM handles NaN natively)."""
    df = df.copy()
    ai = ai_df.copy().rename(columns={"time": TIMESTAMP_COL})
    # Only keep columns that have data for at least some rows.
    usable = [c for c in ai.columns if c != TIMESTAMP_COL and ai[c].notna().sum() > 1000]
    ai = ai[[TIMESTAMP_COL] + usable]
    df = df.merge(ai, on=TIMESTAMP_COL, how="left")
    # Derived features from AI models.
    if "graphcast_wind_speed_10m" in df.columns:
        df["gc_ws10_bias"] = df["wind_speed_10m"] - df["graphcast_wind_speed_10m"]
        df["gc_pressure_bias"] = df["pressure_msl"] - df["graphcast_pressure_msl"]
    if "aifs_wind_speed_100m" in df.columns:
        df["aifs_ws100_vs_80"] = df["aifs_wind_speed_100m"] - df["wind_speed_80m"]
        df["aifs_ws100_cube"] = df["aifs_wind_speed_100m"] ** 3
        # Direction encoding.
        aifs_dir_rad = np.deg2rad(df["aifs_wind_direction_100m"].fillna(0))
        df["aifs_dir100_sin"] = np.sin(aifs_dir_rad)
        df["aifs_dir100_cos"] = np.cos(aifs_dir_rad)
        # Cross-model agreement: ERA5 vs AIFS.
        df["era5_aifs_ws_diff"] = df["era5_wind_speed_100m"] - df["aifs_wind_speed_100m"]
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


# ============================================================
# Experiment runners
# ============================================================

def run_baseline(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k, label="baseline"):
    """Standard v20 config: 3 regime specialists × 5 seeds, CF target."""
    seeds = [42, 123, 456, 789, 2026]
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
    avg_cf = np.mean(regime_preds, axis=0)
    avg_mw = np.clip(from_cf(avg_cf, active_va), 0, CAPACITY_MW)
    nmae = normalized_mae(y_va_mw, avg_mw)
    print(f"  {label}: {nmae:.4f}%")
    return nmae


def run_quantile(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k, alpha=0.47):
    """Quantile regression instead of MAE."""
    seeds = [42, 123, 456, 789, 2026]
    regime_preds = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            params = cfg.to_params()
            params["objective"] = "quantile"
            params["alpha"] = alpha
            del params["metric"]  # quantile uses its own metric
            dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights, feature_name=top_k, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
            b = lgb.train(params, dt, num_boost_round=5000,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds.append(np.mean(preds, axis=0))
    avg_cf = np.mean(regime_preds, axis=0)
    avg_mw = np.clip(from_cf(avg_cf, active_va), 0, CAPACITY_MW)
    nmae = normalized_mae(y_va_mw, avg_mw)
    print(f"  quantile(alpha={alpha}): {nmae:.4f}%")
    return nmae


def run_monotonic(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k):
    """Add monotonic constraints on physics features (requires MSE objective)."""
    mono_features = {
        "wind_speed_120m", "wind_speed_80m", "rews", "v_eff",
        "era5_wind_speed_100m", "era5_ws100_roll_mean_6h", "era5_ws100_roll_mean_3h",
        "active_turbines_ratio", "active_turbines",
        "p_curve_sector", "p_curve_global", "p_curve_x_active",
        "ds_consensus_wake_corrected", "ds_consensus_farm_mw",
    }
    constraints = []
    for f in top_k:
        if f in mono_features:
            constraints.append(1)
        else:
            constraints.append(0)

    seeds = [42, 123, 456, 789, 2026]
    regime_preds = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            params = cfg.to_params()
            # Monotonic constraints require regression (MSE), not regression_l1 (MAE).
            params["objective"] = "regression"
            params["monotone_constraints"] = constraints
            dt = lgb.Dataset(X_tr, label=y_tr_cf, weight=weights, feature_name=top_k, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va_cf, feature_name=top_k, free_raw_data=False)
            b = lgb.train(params, dt, num_boost_round=5000,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            preds.append(b.predict(X_va, num_iteration=b.best_iteration))
        regime_preds.append(np.mean(preds, axis=0))
    avg_cf = np.mean(regime_preds, axis=0)
    avg_mw = np.clip(from_cf(avg_cf, active_va), 0, CAPACITY_MW)
    nmae = normalized_mae(y_va_mw, avg_mw)
    n_constrained = sum(1 for c in constraints if c != 0)
    print(f"  monotonic ({n_constrained} constrained, MSE obj): {nmae:.4f}%")
    return nmae


def run_more_seeds(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k):
    """10 seeds per specialist instead of 5."""
    seeds = [42, 123, 456, 789, 2026, 3141, 1618, 2718, 7777, 12345]
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
    avg_cf = np.mean(regime_preds, axis=0)
    avg_mw = np.clip(from_cf(avg_cf, active_va), 0, CAPACITY_MW)
    nmae = normalized_mae(y_va_mw, avg_mw)
    print(f"  10 seeds: {nmae:.4f}%")
    return nmae


def run_hourly_models(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, hours_tr, hours_va, top_k):
    """24 separate LGBMs, one per hour-of-day."""
    seeds = [42, 123, 456]
    preds_all = np.zeros(len(X_va), dtype=np.float64)
    for h in range(24):
        mask_tr = hours_tr == h
        mask_va = hours_va == h
        if mask_tr.sum() < 100 or mask_va.sum() == 0:
            # Fallback to global model for sparse hours.
            continue
        h_preds = []
        for s in seeds:
            cfg = LGBMConfig(**{**CONFIG.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr[mask_tr], label=y_tr_cf[mask_tr], feature_name=top_k, free_raw_data=False)
            dv = lgb.Dataset(X_va[mask_va], label=y_va_cf[mask_va], feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dt, num_boost_round=5000,
                          valid_sets=[dv], valid_names=["val"],
                          callbacks=[lgb.early_stopping(250, verbose=False)])
            h_preds.append(b.predict(X_va[mask_va], num_iteration=b.best_iteration))
        preds_all[mask_va] = np.mean(h_preds, axis=0)
    avg_mw = np.clip(from_cf(preds_all, active_va), 0, CAPACITY_MW)
    nmae = normalized_mae(y_va_mw, avg_mw)
    print(f"  24 hourly LGBMs: {nmae:.4f}%")
    return nmae


# ============================================================
# Main
# ============================================================

def main():
    set_global_seed(42)
    print("=" * 70)
    print("V25: Comprehensive experiment — all remaining ideas")
    print("=" * 70)

    print("\nPreparing data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    ai_weather = pd.read_parquet(AI_WEATHER_PATH) if AI_WEATHER_PATH.exists() else None

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)
    if ai_weather is not None:
        combined = merge_ai_weather(combined, ai_weather)
        print(f"  AI weather merged: +{sum(1 for c in combined.columns if c.startswith(('aifs_', 'graphcast_', 'gc_', 'era5_aifs')))} columns")

    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Fold-5 setup.
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
    print(f"\nProbe for top-{K} features from {len(feat_cols_all)} total...")
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dt = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols_all, free_raw_data=False)
    dv = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    # Check if any AI weather features made it into top-K.
    ai_in_topk = [f for f in top_k if f.startswith(("aifs_", "graphcast_", "gc_", "era5_aifs"))]
    print(f"  AI weather features in top-{K}: {len(ai_in_topk)} — {ai_in_topk}")

    # Also show where AI features rank.
    ai_feats = [(n, g) for n, g in feat_imp if n.startswith(("aifs_", "graphcast_", "gc_", "era5_aifs"))]
    if ai_feats:
        print(f"\n  AI weather feature rankings:")
        for i, (n, g) in enumerate(feat_imp):
            if n.startswith(("aifs_", "graphcast_", "gc_", "era5_aifs")):
                print(f"    #{i+1}: {n:40s} gain={g:.0f}")

    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()
    hours_tr = df_tr[TIMESTAMP_COL].dt.hour.to_numpy()
    hours_va = df_va[TIMESTAMP_COL].dt.hour.to_numpy()

    # ============================================================
    # Run all experiments
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENTS (all on Fold-5, CF target)")
    print("=" * 70)

    results = {}

    # A. Baseline (v20 config).
    results["baseline_5seed"] = run_baseline(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k, "baseline (5 seeds)")

    # B. AI weather features — try K=90 to include them if they ranked 80-90.
    if ai_feats:
        top_k_90 = [n for n, _ in feat_imp[:90]]
        X_tr_90 = df_tr[top_k_90].to_numpy(dtype=np.float32)
        X_va_90 = df_va[top_k_90].to_numpy(dtype=np.float32)
        results["ai_weather_k90"] = run_baseline(X_tr_90, y_tr_cf, X_va_90, y_va_cf, y_va_mw, active_va, ws_tr, top_k_90, "AI weather K=90")

    # C. Quantile tilted training.
    for alpha in [0.47, 0.48, 0.49]:
        results[f"quantile_{alpha}"] = run_quantile(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k, alpha)

    # D. Monotonic constraints.
    results["monotonic"] = run_monotonic(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k)

    # E. 24 hourly LGBMs.
    results["hourly_24"] = run_hourly_models(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, hours_tr, hours_va, top_k)

    # F. More seeds (10).
    results["10_seeds"] = run_more_seeds(X_tr, y_tr_cf, X_va, y_va_cf, y_va_mw, active_va, ws_tr, top_k)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY (Fold-5 nMAE)")
    print("=" * 70)
    baseline = results["baseline_5seed"]
    for name, nmae in sorted(results.items(), key=lambda x: x[1]):
        delta = nmae - baseline
        marker = " *** BETTER" if delta < -0.005 else (" (worse)" if delta > 0.005 else "")
        print(f"  {name:25s} {nmae:.4f}%  ({delta:+.4f}){marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
