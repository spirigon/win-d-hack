"""ERA5 FINAL: Physics features + ERA5 reanalysis + clean power curve + multi-seed.

This is the new best model. ERA5 reanalysis from Open-Meteo provides an
independent, higher-quality estimate of actual weather conditions. Adding
ERA5 100m wind speed and bias features drops Fold-5 from 8.76% to 8.15%.

Usage:
    python -m src.training.train_era5_final
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
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v2.0_era5_final.csv"

V_CUT_IN = 3.0
V_CUT_OUT = 25.0
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


def merge_era5(df: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")

    # Bias features.
    df["ws10_bias"] = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"] = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"] = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]

    # ERA5 derived.
    df["era5_ws100_cube"] = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]

    # ERA5 direction (degrees → sin/cos).
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)

    # Fill any NaN.
    era5_new_cols = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias") or c == "ws100_vs_80"]
    df[era5_new_cols] = df[era5_new_cols].fillna(0)
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


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    print(f"  Train: {len(df_train)}, Valid: {len(df_valid)}, ERA5: {len(era5)}")

    impossible = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = impossible.values

    print("Building features + merging ERA5...")
    df_train = build_features(df_train, sort_by_time=False)
    df_train = merge_era5(df_train, era5)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)
    df_valid_sorted = merge_era5(df_valid_sorted, era5)

    config_base = LGBMConfig(
        num_leaves=207, min_data_in_leaf=235, learning_rate=0.02221,
        feature_fraction=0.947, bagging_fraction=0.714, bagging_freq=6,
        lambda_l1=0.00192, lambda_l2=2.059,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    # === 5-fold CV ===
    print("\n=== 5-fold Walk-forward CV (ERA5) ===")
    folds = default_folds()
    fold_results = []

    for fold in folds:
        train_idx, val_idx = split_indices(df_train, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        fold_train = df_train.iloc[train_idx]
        fit_data = fold_train[~fold_train["_is_impossible"]]
        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

        df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
        df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
        feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]

        X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
        y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        v_eff_va = df_va["v_eff"].to_numpy()

        booster = _train_single(X_tr, y_tr, X_va, y_va, feat_cols, 42, config_base)
        preds = _post_process(np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW), v_eff_va)
        nmae = normalized_mae(y_va, preds)
        fold_results.append(nmae)
        print(f"  {fold.name}: nMAE = {nmae:.4f} %  (best_iter={booster.best_iteration})")

    print(f"\n  Mean 5-fold nMAE: {np.mean(fold_results):.4f} %")

    # === Fold-5 multi-seed ensemble ===
    print(f"\n=== Fold-5 multi-seed ensemble ({len(SEEDS)} seeds) ===")
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    fold_train = df_train.iloc[train_idx]
    fit_data = fold_train[~fold_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
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
        print(f"  Seed {seed}: nMAE = {nmae:.4f} %  (best_iter={booster.best_iteration})")

    preds_ens = np.mean(preds_all, axis=0)
    nmae_ens = normalized_mae(y_va, preds_ens)
    print(f"\n  >>> Ensemble Fold-5 nMAE: {nmae_ens:.4f} % <<<")

    # === Full-fit ===
    print(f"\n=== Full-fit ({len(SEEDS)} seeds) ===")
    df_train_clean = df_train[~df_train["_is_impossible"]]
    pc_sector_full = fit_sector_isotonic(df_train_clean, n_sectors=8)
    pc_global_full = IsotonicPowerCurve().fit(df_train_clean["v_eff"], df_train_clean[TARGET_COL])
    df_train_full = _add_power_curve_features(df_train, pc_sector_full, pc_global_full)
    feat_cols = [c for c in feature_columns(df_train_full) if c != "_is_impossible"]
    print(f"  Features: {len(feat_cols)}")

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    n_rounds = max(int(np.median(best_iters) * 1.2), 1000)
    print(f"  n_rounds per seed: {n_rounds}")
    boosters_full = [_train_full(X_full, y_full, feat_cols, s, n_rounds, config_base) for s in SEEDS]

    # === Predict Q1-2026 ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_sector_full, pc_global_full)
    for col in set(feat_cols) - set(df_valid_pred.columns):
        df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)
    v_eff_valid = df_valid_pred["v_eff"].to_numpy()

    valid_preds_list = []
    for b in boosters_full:
        p = _post_process(np.clip(b.predict(X_valid), 0, CAPACITY_MW), v_eff_valid)
        valid_preds_list.append(p)

    preds_final = np.mean(valid_preds_list, axis=0)
    preds_final = np.clip(preds_final, 0, CAPACITY_MW)

    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    preds_ordered = np.empty_like(preds_final)
    preds_ordered[order] = preds_final

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))

    print(f"\n  Prediction stats: mean={preds_final.mean():.2f}, std={preds_final.std():.2f}")
    print(f"  P10={np.percentile(preds_final, 10):.2f}, P50={np.median(preds_final):.2f}, P90={np.percentile(preds_final, 90):.2f}")

    # Feature importance.
    imp = boosters_full[0].feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    print("\n=== Top 20 features (gain) ===")
    for name, score in feat_imp[:20]:
        print(f"  {name:40s} {score:12.1f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(boosters_full):
        b.save_model(str(MODEL_DIR / f"lgbm_era5_seed{SEEDS[i]}.txt"))
    print(f"\n  Models saved to {MODEL_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
