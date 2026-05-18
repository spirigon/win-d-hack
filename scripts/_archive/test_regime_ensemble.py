"""Test regime-aware ensemble: train models on different wind-speed regimes.

Paper §8.1: detect meteorological regimes and apply specialized submodels.

Strategy: partition training data by wind speed regime, train specialist
models, then at inference route each prediction hour to the right model
based on its wind speed. Blend 2+ models at regime boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
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

ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
SEEDS = [42, 123, 456, 789, 2026]


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


def train_seeds(X_tr, y_tr, X_va, y_va, feat_cols, w_tr, config):
    preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        if w_tr is not None:
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols, free_raw_data=False)
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        p = np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW)
        preds.append(p)
    return np.mean(preds, axis=0)


def main():
    set_global_seed(42)
    df_train = load_train(_ROOT / "data" / "raw" / "train_dataset.csv")
    df_valid = load_valid_features(_ROOT / "data" / "raw" / "valid_features.csv")
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
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]
    config = LGBMConfig(
        num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
        feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
        lambda_l1=0.253, lambda_l2=0.00971,
        num_boost_round=5000, early_stopping_rounds=250, log_period=0,
    )

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    # Probe for top-70.
    cfg0 = LGBMConfig(**{**config.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dtrain, num_boost_round=5000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_70 = [n for n, _ in feat_imp[:70]]

    X_tr = df_tr[top_70].to_numpy(dtype=np.float32)
    X_va = df_va[top_70].to_numpy(dtype=np.float32)

    # Wind speed at 120m for regime.
    ws_tr = df_tr["wind_speed_120m"].to_numpy()
    ws_va = df_va["wind_speed_120m"].to_numpy()

    # Baseline.
    print("=== Baseline (all data) ===")
    preds_base = train_seeds(X_tr, y_tr, X_va, y_va, top_70, None, config)
    nmae_base = normalized_mae(y_va, preds_base)
    print(f"  nMAE: {nmae_base:.4f}%")

    # Strategy: train 3 regime specialists, weighted average based on wind speed.
    # Regime 1: low wind (0-6), focus on cut-in zone.
    # Regime 2: mid wind (5-11), transition/ramp zone (most error).
    # Regime 3: high wind (9+), rated zone.
    # Use overlapping ranges so specialists cover transition zones.
    regimes = [
        ("low_0_7", (0, 7)),
        ("mid_4_12", (4, 12)),
        ("high_8_25", (8, 25)),
    ]

    specialist_preds = {}
    specialist_val_nmae = {}
    for name, (lo, hi) in regimes:
        # Heavy weights for rows in regime, lower weights outside.
        mask_in_regime = (ws_tr >= lo) & (ws_tr < hi)
        n_in = mask_in_regime.sum()
        weights = np.where(mask_in_regime, 2.0, 0.3).astype(np.float32)

        preds = train_seeds(X_tr, y_tr, X_va, y_va, top_70, weights, config)
        specialist_preds[name] = preds

        # Evaluate on validation rows in the regime.
        val_mask = (ws_va >= lo) & (ws_va < hi)
        if val_mask.sum() > 0:
            nmae_in = normalized_mae(y_va[val_mask], preds[val_mask])
        else:
            nmae_in = float("nan")
        nmae_full = normalized_mae(y_va, preds)
        specialist_val_nmae[name] = (nmae_in, nmae_full)
        print(f"\n{name} (ws in [{lo},{hi}), train rows: {n_in}):")
        print(f"  nMAE on regime rows ({val_mask.sum()}): {nmae_in:.4f}%")
        print(f"  nMAE full: {nmae_full:.4f}%")

    # Combine: route each valid row to best specialist.
    # Soft combination: sigmoid gate between regimes.
    preds_routed = np.zeros_like(ws_va)
    # Use: baseline for transition zones, specialist for extremes.

    # Simple routing: use specialist matching the row's ws.
    # low for ws<5, mid for 5<=ws<9, high for ws>=9.
    mask_low = ws_va < 5
    mask_mid = (ws_va >= 5) & (ws_va < 9)
    mask_high = ws_va >= 9
    preds_routed[mask_low] = specialist_preds["low_0_7"][mask_low]
    preds_routed[mask_mid] = specialist_preds["mid_4_12"][mask_mid]
    preds_routed[mask_high] = specialist_preds["high_8_25"][mask_high]
    nmae_routed = normalized_mae(y_va, preds_routed)
    print(f"\nRouted (hard regime selection): {nmae_routed:.4f}%")

    # Smooth: average of all 3 specialists.
    preds_avg3 = np.mean(list(specialist_preds.values()), axis=0)
    nmae_avg3 = normalized_mae(y_va, preds_avg3)
    print(f"Average of 3 specialists: {nmae_avg3:.4f}%")

    # Blend baseline + average of 3 specialists.
    preds_mix = 0.5 * preds_base + 0.5 * preds_avg3
    nmae_mix = normalized_mae(y_va, preds_mix)
    print(f"Baseline + avg3 (50/50): {nmae_mix:.4f}%")

    # Blend baseline + routed.
    preds_mix2 = 0.5 * preds_base + 0.5 * preds_routed
    nmae_mix2 = normalized_mae(y_va, preds_mix2)
    print(f"Baseline + routed (50/50): {nmae_mix2:.4f}%")

    print("\n=== SUMMARY ===")
    print(f"  Baseline:               {nmae_base:.4f}%")
    print(f"  Routed specialist:      {nmae_routed:.4f}%  ({nmae_routed-nmae_base:+.4f})")
    print(f"  Avg of 3 specialists:   {nmae_avg3:.4f}%  ({nmae_avg3-nmae_base:+.4f})")
    print(f"  Base + avg3 (50/50):    {nmae_mix:.4f}%  ({nmae_mix-nmae_base:+.4f})")
    print(f"  Base + routed (50/50):  {nmae_mix2:.4f}%  ({nmae_mix2-nmae_base:+.4f})")


if __name__ == "__main__":
    main()
