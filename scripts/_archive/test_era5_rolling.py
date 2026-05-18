"""Test ERA5 rolling/temporal features on top of v8 wake setup.

ERA5 is higher quality than NWP. Rolling stats on ERA5 might capture
temporal weather patterns that help (unlike NWP rolling which hurt before).

Features to test:
- ERA5 ws100 rolling mean/std over 3/6/12/24h
- ERA5 ws100 diff (ramp rate)
- ERA5 pressure tendency
- ERA5 direction change rate
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
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TURBINES_IN_MAINTENANCE_COL
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
V_CUT_IN = 3.0
SEEDS = [42, 123, 456, 789, 2026]


def _add_power_curve_features(df, pc_sector, pc_global):
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
    """Add ERA5 rolling/temporal features. Must be sorted chronologically."""
    df = df.copy()
    ws = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]

    # Rolling stats on ERA5 100m wind.
    for w in [3, 6, 12, 24]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"] = roll.std().fillna(0)

    # ERA5 wind ramp (diff).
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)

    # ERA5 pressure tendency.
    df["era5_pressure_diff3"] = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"] = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)

    # ERA5 turbulence intensity.
    roll6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)

    # ERA5 direction change.
    dir_sin = df["era5_dir100_sin"]
    dir_cos = df["era5_dir100_cos"]
    df["era5_dir_sin_diff1"] = dir_sin.diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = dir_cos.diff(1).fillna(0)

    return df


def main():
    set_global_seed(42)
    df_train = load_train(_ROOT / "data" / "raw" / "train_dataset.csv")
    df_valid = load_valid_features(_ROOT / "data" / "raw" / "valid_features.csv")
    era5 = pd.read_parquet(ERA5_PATH)

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    # Build features: concatenate train+valid for rolling continuity.
    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)

    # Split back.
    df_train = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)

    # Re-identify impossible rows (lost during concat).
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    config = LGBMConfig(
        num_leaves=453, min_data_in_leaf=121, learning_rate=0.01555,
        feature_fraction=0.695, bagging_fraction=0.509, bagging_freq=5,
        lambda_l1=0.00294, lambda_l2=0.342,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)
    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols_all = [c for c in feature_columns(df_tr) if c != "_is_impossible" and c != "_split"]
    print(f"Total features (with ERA5 rolling): {len(feat_cols_all)}")

    # Identify new ERA5 rolling features.
    era5_roll_cols = [c for c in feat_cols_all if "era5_ws100_roll" in c or "era5_ws100_diff" in c
                     or "era5_pressure_diff" in c or "era5_turb" in c or "era5_dir_sin_diff" in c
                     or "era5_dir_cos_diff" in c]
    print(f"New ERA5 rolling features: {len(era5_roll_cols)}")

    X_tr_all = df_tr[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    # Probe for importance.
    cfg = LGBMConfig(**{**config.__dict__, "seed": 42})
    dtrain = lgb.Dataset(X_tr_all, label=y_tr, feature_name=feat_cols_all, free_raw_data=False)
    dval = lgb.Dataset(X_va_all, label=y_va, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])

    print("\nERA5 rolling features in importance ranking:")
    for rank, (name, score) in enumerate(feat_imp, 1):
        if name in era5_roll_cols:
            print(f"  rank {rank:3d}: {name:40s} {score:10.1f}")

    # K ablation.
    print("\n=== K ablation (5 seeds) ===")
    for k in [50, 60, 70, 80, len(feat_cols_all)]:
        top_k = [n for n, _ in feat_imp[:k]]
        X_tr_k = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va_k = df_va[top_k].to_numpy(dtype=np.float32)
        preds_all = []
        for seed in SEEDS:
            cfg = LGBMConfig(**{**config.__dict__, "seed": seed})
            dtrain = lgb.Dataset(X_tr_k, label=y_tr, feature_name=top_k, free_raw_data=False)
            dval = lgb.Dataset(X_va_k, label=y_va, feature_name=top_k, free_raw_data=False)
            b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                          valid_sets=[dval], valid_names=["val"],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            p = np.clip(b.predict(X_va_k, num_iteration=b.best_iteration), 0, CAPACITY_MW)
            preds_all.append(p)
        ens = np.mean(preds_all, axis=0)
        nmae = normalized_mae(y_va, ens)
        # Count how many ERA5 rolling features are in top_k.
        n_roll = len([c for c in top_k if c in era5_roll_cols])
        print(f"  K={k:3d}: nMAE={nmae:.4f}% ({n_roll} ERA5-rolling in top-K)")

    # Reference: v8 without ERA5 rolling (K=60).
    print("\n  Reference v8 (no ERA5 rolling, K=60): ~7.97%")


if __name__ == "__main__":
    main()
