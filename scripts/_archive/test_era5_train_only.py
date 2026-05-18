"""Test: ERA5 on training only, NOT on valid (conservative approach).

If ERA5 is only used during training, the model learns better weather-power
relationships but at inference it only sees the original NWP features.
The ERA5 columns will be missing at inference → set to 0 or drop them.

Two strategies:
A) Train WITH ERA5 features, set them to 0 at inference (model learns to
   use them when available but falls back to other features).
B) Train WITHOUT ERA5 features but use ERA5 to REPLACE the original weather
   columns during training (model learns from better data, infers on NWP).
C) Train with ERA5 features, at inference fill ERA5 columns with the
   corresponding NWP values (approximate substitution).

Strategy C is most promising: at inference, era5_wind_speed_100m ≈ wind_speed_80m,
era5_pressure_msl ≈ pressure_msl, etc. The model learns the relationship
between ERA5 and power, and at inference we feed it the NWP values as a proxy.
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
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

V_CUT_IN = 3.0
V_CUT_OUT = 25.0
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
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


def merge_era5_full(df, era5):
    """Merge ERA5 with all derived features."""
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


def fill_era5_from_nwp(df):
    """Strategy C: fill ERA5 columns with NWP proxies at inference time."""
    df = df.copy()
    # Map ERA5 columns to their closest NWP equivalents.
    df["era5_wind_speed_10m"] = df["wind_speed_10m"]
    df["era5_wind_speed_100m"] = df["wind_speed_80m"]  # 100m ≈ 80m
    df["era5_wind_direction_10m"] = df["wind_direction_10m"] * 1000.0  # convert to degrees
    df["era5_wind_direction_100m"] = df["wind_direction_80m"] * 1000.0
    df["era5_wind_gusts_10m"] = df["wind_gusts_10m"]
    df["era5_temperature_2m"] = df["temperature_80m"]  # rough proxy
    df["era5_pressure_msl"] = df["pressure_msl"]
    df["era5_cloud_cover_low"] = df["cloud_cover_low"] * 100.0  # our data is 0-1, ERA5 is 0-100%
    df["era5_rain"] = df["rain"]
    df["era5_snowfall"] = df["snowfall"]
    # Derived.
    df["ws10_bias"] = 0.0  # no bias when using same source
    df["ws100_vs_80"] = 0.0
    df["gust_bias"] = 0.0
    df["pressure_bias"] = 0.0
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
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

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    df_train = build_features(df_train, sort_by_time=False)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    # Train: merge real ERA5.
    df_train = merge_era5_full(df_train, era5)

    config = LGBMConfig(
        num_leaves=207, min_data_in_leaf=235, learning_rate=0.02221,
        feature_fraction=0.947, bagging_fraction=0.714, bagging_freq=6,
        lambda_l1=0.00192, lambda_l2=2.059,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va_real = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    y_va = df_va_real[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va_real["v_eff"].to_numpy()

    print(f"Features: {len(feat_cols)}")

    # === Strategy A: ERA5 on val = 0 ===
    print("\n=== Strategy A: ERA5 columns set to 0 at inference ===")
    df_va_zero = df_va_real.copy()
    era5_cols = [c for c in feat_cols if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    for c in era5_cols:
        if c in df_va_zero.columns:
            df_va_zero[c] = 0.0
    X_va_zero = df_va_zero[feat_cols].to_numpy(dtype=np.float32)

    preds_a = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        # Use real ERA5 val for early stopping (model sees ERA5 during training).
        X_va_real = df_va_real[feat_cols].to_numpy(dtype=np.float32)
        dval = lgb.Dataset(X_va_real, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va_zero, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        preds_a.append(p)
    ens_a = np.mean(preds_a, axis=0)
    print(f"  Strategy A ensemble nMAE: {normalized_mae(y_va, ens_a):.4f} %")

    # === Strategy C: ERA5 filled from NWP proxies at inference ===
    print("\n=== Strategy C: ERA5 columns filled from NWP proxies ===")
    df_va_proxy = df_train.iloc[val_idx].copy()
    # Remove real ERA5 and replace with NWP proxies.
    era5_raw_cols = [c for c in df_va_proxy.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df_va_proxy = df_va_proxy.drop(columns=era5_raw_cols, errors="ignore")
    df_va_proxy = fill_era5_from_nwp(df_va_proxy)
    df_va_proxy = _add_power_curve_features(df_va_proxy, pc_sector, pc_global)
    # Ensure all feat_cols exist.
    for c in feat_cols:
        if c not in df_va_proxy.columns:
            df_va_proxy[c] = 0.0
    X_va_proxy = df_va_proxy[feat_cols].to_numpy(dtype=np.float32)

    preds_c = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va_real, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va_proxy, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        preds_c.append(p)
    ens_c = np.mean(preds_c, axis=0)
    print(f"  Strategy C ensemble nMAE: {normalized_mae(y_va, ens_c):.4f} %")

    # === Reference: ERA5 on both (what we already have) ===
    print("\n=== Reference: ERA5 on both train and val ===")
    preds_ref = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va_real, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va_real, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        preds_ref.append(p)
    ens_ref = np.mean(preds_ref, axis=0)
    print(f"  Reference (ERA5 both) ensemble nMAE: {normalized_mae(y_va, ens_ref):.4f} %")

    # === Baseline: no ERA5 at all ===
    print("\n=== Baseline: no ERA5 features ===")
    feat_cols_base = [c for c in feat_cols if not c.startswith("era5_") and not c.endswith("_bias") and c != "ws100_vs_80"]
    X_tr_base = df_tr[feat_cols_base].to_numpy(dtype=np.float32)
    X_va_base = df_va_real[feat_cols_base].to_numpy(dtype=np.float32)
    preds_base = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr_base, label=y_tr, feature_name=feat_cols_base, free_raw_data=False)
        dval = lgb.Dataset(X_va_base, label=y_va, feature_name=feat_cols_base, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va_base, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        preds_base.append(p)
    ens_base = np.mean(preds_base, axis=0)
    print(f"  Baseline (no ERA5) ensemble nMAE: {normalized_mae(y_va, ens_base):.4f} %")

    print("\n=== SUMMARY ===")
    print(f"  Baseline (no ERA5):          {normalized_mae(y_va, ens_base):.4f} %")
    print(f"  Strategy A (ERA5=0 at val):  {normalized_mae(y_va, ens_a):.4f} %")
    print(f"  Strategy C (ERA5=NWP proxy): {normalized_mae(y_va, ens_c):.4f} %")
    print(f"  Reference (ERA5 on both):    {normalized_mae(y_va, ens_ref):.4f} %")


if __name__ == "__main__":
    main()
