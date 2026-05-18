"""Test two new ideas from literature:
1. Capacity-normalized target (CF = P / P_available)
2. Regime cluster labels as features (KMeans on shear + temp gradient + ws)
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TURBINES_IN_MAINTENANCE_COL, TOTAL_TURBINES
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
TURBINE_RATED_MW = 3.465


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


def add_regime_labels(df, kmeans_model=None, n_clusters=5):
    """Add KMeans regime cluster labels based on atmospheric state."""
    features_for_cluster = [
        "hellmann_alpha",
        "wind_speed_120m",
        "air_density",
    ]
    # Add temp gradient if available.
    if "temp_gradient" in df.columns:
        features_for_cluster.append("temp_gradient")

    X_cluster = df[features_for_cluster].fillna(0).to_numpy()

    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(X_cluster)

    labels = kmeans_model.predict(X_cluster)
    df = df.copy()
    df["regime_cluster"] = labels.astype(np.int8)
    return df, kmeans_model


def train_seeds(X_tr, y_tr, X_va, y_va, feat_cols, config):
    preds = []
    for s in SEEDS:
        cfg = LGBMConfig(**{**config.__dict__, "seed": s})
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
    df_valid_sorted = combined[combined["_split"] == "valid"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    config = LGBMConfig(
        num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
        feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
        lambda_l1=0.253, lambda_l2=0.00971,
        num_boost_round=5000, early_stopping_rounds=250, log_period=0,
    )

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

    # === Test 1: Regime cluster labels ===
    print("=== Test 1: Regime cluster labels ===")
    df_tr_r, km = add_regime_labels(df_tr, n_clusters=5)
    df_va_r, _ = add_regime_labels(df_va, kmeans_model=km)

    feat_cols_all = [c for c in feature_columns(df_tr_r) if c not in ("_is_impossible", "_split")]
    X_tr_all = df_tr_r[feat_cols_all].to_numpy(dtype=np.float32)
    y_tr = df_tr_r[TARGET_COL].to_numpy(dtype=np.float32)
    X_va_all = df_va_r[feat_cols_all].to_numpy(dtype=np.float32)
    y_va = df_va_r[TARGET_COL].to_numpy(dtype=np.float32)

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

    # Check where regime_cluster ranks.
    for rank, (name, score) in enumerate(feat_imp, 1):
        if "regime" in name:
            print(f"  regime_cluster rank: {rank}, importance: {score:.1f}")
            break

    X_tr = df_tr_r[top_70].to_numpy(dtype=np.float32)
    X_va = df_va_r[top_70].to_numpy(dtype=np.float32)
    preds_regime = train_seeds(X_tr, y_tr, X_va, y_va, top_70, config)
    nmae_regime = normalized_mae(y_va, preds_regime)
    print(f"  With regime label: {nmae_regime:.4f}%")

    # Baseline without regime label.
    feat_no_regime = [c for c in top_70 if "regime" not in c]
    if len(feat_no_regime) < 70:
        # Add next features from importance list.
        extra = [n for n, _ in feat_imp if n not in top_70 and "regime" not in n][:70 - len(feat_no_regime)]
        feat_no_regime.extend(extra)
    X_tr_nr = df_tr_r[feat_no_regime].to_numpy(dtype=np.float32)
    X_va_nr = df_va_r[feat_no_regime].to_numpy(dtype=np.float32)
    preds_no_regime = train_seeds(X_tr_nr, y_tr, X_va_nr, y_va, feat_no_regime, config)
    nmae_no_regime = normalized_mae(y_va, preds_no_regime)
    print(f"  Without regime label: {nmae_no_regime:.4f}%")
    print(f"  Delta: {nmae_regime - nmae_no_regime:+.4f}")

    # === Test 2: Capacity-normalized target ===
    print("\n=== Test 2: Capacity-normalized target ===")
    # CF = P / P_available where P_available = active_turbines × 3.465 MW.
    active_tr = df_tr[TURBINES_IN_MAINTENANCE_COL].to_numpy()
    active_va = df_va[TURBINES_IN_MAINTENANCE_COL].to_numpy()
    p_avail_tr = (TOTAL_TURBINES - active_tr) * TURBINE_RATED_MW
    p_avail_va = (TOTAL_TURBINES - active_va) * TURBINE_RATED_MW

    y_tr_cf = y_tr / p_avail_tr  # capacity factor (0-1 range)
    y_va_cf = y_va / p_avail_va

    print(f"  CF stats: mean={y_tr_cf.mean():.4f}, max={y_tr_cf.max():.4f}")

    # Train on CF, predict CF, then reverse-transform.
    preds_cf = train_seeds(X_tr_nr, y_tr_cf, X_va_nr, y_va_cf, feat_no_regime, config)
    # Reverse: P = CF × P_available.
    preds_mw = preds_cf * p_avail_va
    preds_mw = np.clip(preds_mw, 0, CAPACITY_MW)
    nmae_cf = normalized_mae(y_va, preds_mw)
    print(f"  CF-normalized training: {nmae_cf:.4f}%")
    print(f"  Baseline (raw MW):      {nmae_no_regime:.4f}%")
    print(f"  Delta: {nmae_cf - nmae_no_regime:+.4f}")


if __name__ == "__main__":
    main()
