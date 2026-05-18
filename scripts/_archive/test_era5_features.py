"""Test ERA5 features: merge with training data, compute bias, evaluate.

ERA5 provides 10m and 100m wind. Our training data has 10m/80m/120m/180m.
Useful features:
1. era5_ws_100m — independent estimate of hub-height wind (100m ≈ 80m hub).
2. era5_ws_10m_bias = train_ws_10m - era5_ws_10m — NWP forecast error.
3. era5_ws_100m_cube — cubic for power.
4. era5_gust_10m — independent gust estimate.
5. era5_pressure_msl — independent pressure.
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


def _post_process(preds, v_eff):
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def merge_era5(df: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """Merge ERA5 features into the main dataframe by timestamp."""
    df = df.copy()
    era5 = era5.copy()
    era5 = era5.rename(columns={"time": TIMESTAMP_COL})
    # Prefix ERA5 columns.
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")

    # Compute bias features.
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]

    # ERA5 derived.
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]

    # Direction from ERA5 (in degrees, 0-360).
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)

    return df


def main():
    set_global_seed(42)
    TRAIN = _ROOT / "data" / "raw" / "train_dataset.csv"
    VALID = _ROOT / "data" / "raw" / "valid_features.csv"

    print("Loading data...")
    df_train = load_train(TRAIN)
    df_valid = load_valid_features(VALID)
    era5 = pd.read_parquet(ERA5_PATH)
    print(f"  Train: {len(df_train)}, Valid: {len(df_valid)}, ERA5: {len(era5)}")

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values
    df_train = build_features(df_train, sort_by_time=False)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    # Merge ERA5.
    print("Merging ERA5...")
    df_train = merge_era5(df_train, era5)
    df_valid_sorted = merge_era5(df_valid_sorted, era5)

    # Check merge quality.
    era5_cols = [c for c in df_train.columns if c.startswith("era5_") or c.endswith("_bias")]
    n_null = df_train[era5_cols].isna().sum()
    print(f"  ERA5 NaN per column:\n{n_null[n_null > 0]}")
    # Fill any NaN ERA5 with 0 (shouldn't happen for train/valid period).
    df_train[era5_cols] = df_train[era5_cols].fillna(0)
    df_valid_sorted[era5_cols] = df_valid_sorted[era5_cols].fillna(0)

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
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
    print(f"\n  Features: {len(feat_cols)}")

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    # === Baseline (no ERA5) ===
    print("\n=== Baseline (no ERA5 features) ===")
    feat_cols_base = [c for c in feat_cols if not c.startswith("era5_") and not c.endswith("_bias") and c not in ("ws100_vs_80",)]
    X_tr_base = df_tr[feat_cols_base].to_numpy(dtype=np.float32)
    X_va_base = df_va[feat_cols_base].to_numpy(dtype=np.float32)

    SEEDS = [42, 123, 456, 789, 2026]
    base_preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr_base, label=y_tr, feature_name=feat_cols_base, free_raw_data=False)
        dval = lgb.Dataset(X_va_base, label=y_va, feature_name=feat_cols_base, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va_base, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        base_preds.append(p)
    base_ens = np.mean(base_preds, axis=0)
    print(f"  Baseline ensemble nMAE: {normalized_mae(y_va, base_ens):.4f} %")

    # === With ERA5 features ===
    print("\n=== With ERA5 features ===")
    era5_preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        b = lgb.train(cfg.to_params(), dtrain, num_boost_round=4000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = _post_process(np.clip(b.predict(X_va, num_iteration=b.best_iteration), 0, CAPACITY_MW), v_eff_va)
        era5_preds.append(p)
        nmae = normalized_mae(y_va, p)
        print(f"  Seed {s}: nMAE = {nmae:.4f} %  (best_iter={b.best_iteration})")
    era5_ens = np.mean(era5_preds, axis=0)
    print(f"\n  ERA5 ensemble nMAE: {normalized_mae(y_va, era5_ens):.4f} %")

    # Feature importance for ERA5 features.
    imp = b.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    print("\n  ERA5-related features in top 30:")
    for name, score in feat_imp[:30]:
        if "era5" in name or "bias" in name or "ws100_vs" in name:
            print(f"    {name:40s} {score:12.1f}")

    # Delta.
    delta = normalized_mae(y_va, era5_ens) - normalized_mae(y_va, base_ens)
    print(f"\n  Delta (ERA5 - baseline): {delta:+.4f} pp")
    if delta < 0:
        print("  >>> ERA5 features HELP! <<<")
    else:
        print("  >>> ERA5 features do NOT help. <<<")


if __name__ == "__main__":
    main()
