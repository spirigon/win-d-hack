"""Test ECMWF IFS features (9km) on top of ERA5 (25km).

IFS is a different reanalysis at higher resolution. Independent estimate of
same quantities — ensemble diversity.
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
from src.features.era5_features import (
    ERA5SectorPowerCurve,
    add_era5_advanced_features,
    add_era5_power_curve_features,
)
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

V_CUT_IN = 3.0
V_CUT_OUT = 25.0
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
IFS_PATH = _ROOT / "data" / "external" / "ecmwf_ifs.parquet"


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
    return df


def merge_ifs(df, ifs):
    df = df.copy()
    ifs = ifs.copy().rename(columns={"time": TIMESTAMP_COL})
    ifs_cols = [c for c in ifs.columns if c != TIMESTAMP_COL]
    ifs = ifs.rename(columns={c: f"ifs_{c}" for c in ifs_cols})
    df = df.merge(ifs, on=TIMESTAMP_COL, how="left")

    # IFS-vs-NWP biases.
    df["ifs_ws10_bias"] = df["wind_speed_10m"] - df["ifs_wind_speed_10m"]
    df["ifs_ws100_vs_80"] = df["ifs_wind_speed_100m"] - df["wind_speed_80m"]
    df["ifs_gust_bias"] = df["wind_gusts_10m"] - df["ifs_wind_gusts_10m"]
    df["ifs_pressure_bias"] = df["pressure_msl"] - df["ifs_pressure_msl"]

    # IFS derived.
    df["ifs_ws100_cube"] = df["ifs_wind_speed_100m"] ** 3
    df["ifs_ws100_x_active"] = df["ifs_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    ifs_dir_rad = np.deg2rad(df["ifs_wind_direction_100m"])
    df["ifs_dir100_sin"] = np.sin(ifs_dir_rad)
    df["ifs_dir100_cos"] = np.cos(ifs_dir_rad)

    # ERA5-vs-IFS biases (two independent reanalysis disagreement).
    df["era5_vs_ifs_ws100"] = df["era5_wind_speed_100m"] - df["ifs_wind_speed_100m"]
    df["era5_vs_ifs_ws10"] = df["era5_wind_speed_10m"] - df["ifs_wind_speed_10m"]
    df["era5_vs_ifs_temp"] = df["era5_temperature_2m"] - df["ifs_temperature_2m"]
    df["era5_vs_ifs_pressure"] = df["era5_pressure_msl"] - df["ifs_pressure_msl"]

    # Fill NaN.
    new_cols = [c for c in df.columns if c.startswith("ifs_") or "era5_vs_ifs" in c]
    df[new_cols] = df[new_cols].fillna(0)
    return df


def _post_process(preds, v_eff):
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def main():
    set_global_seed(42)

    df_train = load_train(_ROOT / "data" / "raw" / "train_dataset.csv")
    df_valid = load_valid_features(_ROOT / "data" / "raw" / "valid_features.csv")
    era5 = pd.read_parquet(ERA5_PATH)
    ifs = pd.read_parquet(IFS_PATH)
    print(f"Train: {len(df_train)}, Valid: {len(df_valid)}, ERA5: {len(era5)}, IFS: {len(ifs)}")

    # Check IFS data quality.
    print(f"\nIFS sample:\n{ifs.head(3)}")
    print(f"\nIFS NaN counts:\n{ifs.isna().sum()}")

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    df_train = build_features(df_train, sort_by_time=False)

    # Merge both.
    df_train = merge_era5(df_train, era5)
    df_train = merge_ifs(df_train, ifs)
    df_train = add_era5_advanced_features(df_train)
    era5_related = [c for c in df_train.columns if "era5" in c or "_bias" in c or "_vs_" in c or "ws100_vs_80" == c or "ifs_" in c]
    df_train[era5_related] = df_train[era5_related].fillna(0)

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
    era5_pc = ERA5SectorPowerCurve(n_sectors=8).fit(fit_data)

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_tr = add_era5_power_curve_features(df_tr, era5_pc)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    df_va = add_era5_power_curve_features(df_va, era5_pc)
    feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
    print(f"\nFeatures: {len(feat_cols)}")

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    SEEDS = [42, 123, 456, 789, 2026]
    preds_all = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        preds_all.append(p)
        print(f"  Seed {s}: {normalized_mae(y_va, p):.4f}%")
    ens = np.mean(preds_all, axis=0)
    print(f"\n  Ensemble nMAE: {normalized_mae(y_va, ens):.4f}%")

    imp = b.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    print("\n  IFS-related features in top 30:")
    for name, score in feat_imp[:30]:
        if "ifs" in name:
            print(f"    {name:40s} {score:12.1f}")

    # Compare with v3 baseline (ERA5 only).
    print("\n  Reference v3 Fold-5 nMAE: 8.0557 %")


if __name__ == "__main__":
    main()
