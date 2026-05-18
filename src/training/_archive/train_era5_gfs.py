"""V4: ERA5 + GFS ensemble (two independent NWP sources).

Adds GFS 100m wind and basic features on top of v3 advanced ERA5.
Tests if adding a second independent NWP source improves predictions.

Usage:
    python -m src.training.train_era5_gfs
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
from src.features.era5_features import (
    ERA5SectorPowerCurve,
    add_era5_advanced_features,
    add_era5_power_curve_features,
)
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
GFS_PATH = _ROOT / "data" / "external" / "gfs_global.parquet"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v4.0_era5_gfs.csv"

V_CUT_IN = 3.0
V_CUT_OUT = 25.0
SEEDS = [42, 123, 456, 789, 2026, 3141, 1618, 2718, 7777, 12345]


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


def merge_gfs(df, gfs):
    df = df.copy()
    gfs = gfs.copy().rename(columns={"time": TIMESTAMP_COL})
    gfs_cols = [c for c in gfs.columns if c != TIMESTAMP_COL]
    gfs = gfs.rename(columns={c: f"gfs_{c}" for c in gfs_cols})
    df = df.merge(gfs, on=TIMESTAMP_COL, how="left")

    # GFS features.
    df["gfs_ws100_cube"] = df["gfs_wind_speed_100m"] ** 3
    df["gfs_ws100_x_active"] = df["gfs_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    gfs_dir_rad = np.deg2rad(df["gfs_wind_direction_100m"])
    df["gfs_dir100_sin"] = np.sin(gfs_dir_rad)
    df["gfs_dir100_cos"] = np.cos(gfs_dir_rad)

    # Cross-source biases.
    df["gfs_ws100_vs_era5"] = df["gfs_wind_speed_100m"] - df["era5_wind_speed_100m"]
    df["gfs_ws10_vs_nwp"] = df["wind_speed_10m"] - df["gfs_wind_speed_10m"]
    df["gfs_pressure_vs_era5"] = df["gfs_pressure_msl"] - df["era5_pressure_msl"]

    # Consensus wind (mean of all 3 sources at ~100m).
    df["consensus_ws100"] = (df["era5_wind_speed_100m"] + df["gfs_wind_speed_100m"] + df["wind_speed_80m"]) / 3.0
    df["consensus_ws100_cube"] = df["consensus_ws100"] ** 3
    df["consensus_ws100_x_active"] = df["consensus_ws100_cube"] * df["active_turbines_ratio"]

    # Disagreement (std of sources).
    ws_stack = np.stack([df["era5_wind_speed_100m"], df["gfs_wind_speed_100m"], df["wind_speed_80m"]])
    df["ws_source_std"] = ws_stack.std(axis=0)
    df["ws_source_range"] = ws_stack.max(axis=0) - ws_stack.min(axis=0)

    gfs_new = [c for c in df.columns if c.startswith("gfs_") or "consensus" in c or "ws_source" in c]
    df[gfs_new] = df[gfs_new].fillna(0)
    return df


def _post_process(preds, v_eff):
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def _train_single(X_tr, y_tr, X_va, y_va, feat_cols, seed, config_base):
    cfg = LGBMConfig(**{**config_base.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(
        cfg.to_params(), dtrain, num_boost_round=4000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def _train_full(X, y, feat_cols, seed, n_rounds, config_base):
    cfg = LGBMConfig(**{**config_base.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)


def main():
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    gfs = pd.read_parquet(GFS_PATH)
    print(f"  Train: {len(df_train)}, Valid: {len(df_valid)}")

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    print("Building features + merging ERA5 + GFS...")
    df_train = build_features(df_train, sort_by_time=False)
    df_train = merge_era5(df_train, era5)
    df_train = merge_gfs(df_train, gfs)
    df_train = add_era5_advanced_features(df_train)

    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)
    df_valid_sorted = merge_era5(df_valid_sorted, era5)
    df_valid_sorted = merge_gfs(df_valid_sorted, gfs)
    df_valid_sorted = add_era5_advanced_features(df_valid_sorted)

    # Fill any NaN.
    all_ext = [c for c in df_train.columns if ("era5" in c or "gfs" in c or "_bias" in c or "_vs_" in c or "consensus" in c or "ws_source" in c or "ws100_vs_80" == c)]
    df_train[all_ext] = df_train[all_ext].fillna(0)
    df_valid_sorted[all_ext] = df_valid_sorted[all_ext].fillna(0)

    config_base = LGBMConfig(
        num_leaves=453, min_data_in_leaf=121, learning_rate=0.01555,
        feature_fraction=0.695, bagging_fraction=0.509, bagging_freq=5,
        lambda_l1=0.00294, lambda_l2=0.342,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    # === Fold-5 eval ===
    print("\n=== Fold-5 multi-seed ensemble ===")
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
    print(f"  Features: {len(feat_cols)}")

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    preds_all = []
    best_iters = []
    for seed in SEEDS:
        booster = _train_single(X_tr, y_tr, X_va, y_va, feat_cols, seed, config_base)
        p = _post_process(np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW), v_eff_va)
        nmae = normalized_mae(y_va, p)
        preds_all.append(p)
        best_iters.append(booster.best_iteration)
        print(f"  Seed {seed}: {nmae:.4f}%  (iter={booster.best_iteration})")

    preds_ens = np.mean(preds_all, axis=0)
    nmae_ens = normalized_mae(y_va, preds_ens)
    print(f"\n  Ensemble Fold-5: {nmae_ens:.4f}%")

    # === Full-fit ===
    print("\n=== Full-fit ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    era5_pc_full = ERA5SectorPowerCurve(n_sectors=8).fit(df_train_clean)

    df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)
    df_train_full = add_era5_power_curve_features(df_train_full, era5_pc_full)
    feat_cols = [c for c in feature_columns(df_train_full) if c != "_is_impossible"]
    print(f"  Features: {len(feat_cols)}")

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    n_rounds = max(int(np.median(best_iters) * 1.2), 1000)
    boosters_full = [_train_full(X_full, y_full, feat_cols, s, n_rounds, config_base) for s in SEEDS]

    # === Predict ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
    df_valid_pred = add_era5_power_curve_features(df_valid_pred, era5_pc_full)
    for col in set(feat_cols) - set(df_valid_pred.columns):
        df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)
    v_eff_valid = df_valid_pred["v_eff"].to_numpy()

    valid_preds = [_post_process(np.clip(b.predict(X_valid), 0, CAPACITY_MW), v_eff_valid) for b in boosters_full]
    preds_final = np.mean(valid_preds, axis=0)
    preds_final = np.clip(preds_final, 0, CAPACITY_MW)

    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(preds_final)
    po[order] = preds_final
    write_submission(po, SUBMISSION_PATH, expected_rows=len(df_valid))

    # Blend with v3.0 and v2.0.
    v30 = pd.read_csv(_ROOT / "submissions" / "archive" / "v3.0_era5_advanced.csv", header=None)[0].to_numpy()
    v20 = pd.read_csv(_ROOT / "submissions" / "archive" / "v2.0_era5_final.csv", header=None)[0].to_numpy()
    blend3 = np.clip((po + v30 + v20) / 3.0, 0, CAPACITY_MW)
    write_submission(blend3, _ROOT / "submissions" / "archive" / "v4.1_triple_blend.csv", expected_rows=len(df_valid))

    imp = boosters_full[0].feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    print("\n=== Top 20 features (gain) ===")
    for name, score in feat_imp[:20]:
        marker = "*" if "gfs" in name or "consensus" in name or "ws_source" in name else " "
        print(f"  {marker} {name:45s} {score:12.1f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(boosters_full):
        b.save_model(str(MODEL_DIR / f"lgbm_v4_seed{SEEDS[i]}.txt"))
    print("\nDone.")


if __name__ == "__main__":
    main()
