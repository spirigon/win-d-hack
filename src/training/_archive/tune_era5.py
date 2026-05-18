"""Optuna tuning for ERA5 feature set.

The feature space changed from 88 to 106 features with ERA5. The old
hyperparameters (tuned for 88 features) may not be optimal. Retune
specifically for Fold-5 (Q1 surrogate).

Usage:
    python -m src.training.tune_era5 --n-trials 80
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
STUDY_PATH = _ROOT / "models" / "optuna_era5_study.pkl"
CONFIG_OUT = _ROOT / "configs" / "model" / "lgbm_era5_tuned.yaml"

V_CUT_IN = 3.0
V_CUT_OUT = 25.0


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


def _post_process(preds, v_eff):
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def _prepare():
    set_global_seed(42)
    df = load_train(TRAIN_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    impossible = identify_impossible_rows(df)
    df["_is_impossible"] = impossible.values
    df = build_features(df, sort_by_time=False)
    df = merge_era5(df, era5)

    folds = default_folds()
    splits = {}
    for name in ("fold4_2024Q4", "fold5_2025Q1"):
        fold = next(f for f in folds if f.name == name)
        train_idx, val_idx = split_indices(df, fold)
        fold_train = df.iloc[train_idx]
        fit_data = fold_train[~fold_train["_is_impossible"]]
        pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
        df_va = _add_power_curve_features(df.iloc[val_idx], pc_sector, pc_global)
        feat_cols = [c for c in feature_columns(df_tr) if c != "_is_impossible"]
        X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
        y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        v_eff_va = df_va["v_eff"].to_numpy()
        splits[name] = (X_tr, y_tr, X_va, y_va, v_eff_va, feat_cols)
    return splits


def objective(trial, splits):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 63, 511),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.08, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 8),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
    }

    nmaes = {}
    for name, (X_tr, y_tr, X_va, y_va, v_eff_va, feat_cols) in splits.items():
        cfg = LGBMConfig(num_boost_round=2000, early_stopping_rounds=150, log_period=0, **params)
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
        booster = lgb.train(
            cfg.to_params(), dtrain, num_boost_round=2000,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        preds = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW)
        preds = _post_process(preds, v_eff_va)
        nmaes[name] = normalized_mae(y_va, preds)

    trial.set_user_attr("fold4_nmae", nmaes["fold4_2024Q4"])
    trial.set_user_attr("fold5_nmae", nmaes["fold5_2025Q1"])
    return 0.7 * nmaes["fold5_2025Q1"] + 0.3 * nmaes["fold4_2024Q4"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=80)
    args = parser.parse_args()

    print("Preparing splits with ERA5 features...")
    splits = _prepare()
    print(f"  Feature count: {len(splits['fold5_2025Q1'][5])}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def cb(study, trial):
        f5 = trial.user_attrs.get("fold5_nmae", float("nan"))
        f4 = trial.user_attrs.get("fold4_nmae", float("nan"))
        print(f"  trial {trial.number:3d}  obj={trial.value:.4f}  f5={f5:.4f}  f4={f4:.4f}  best={study.best_value:.4f}")

    print(f"\nRunning {args.n_trials} trials...")
    study.optimize(lambda t: objective(t, splits), n_trials=args.n_trials, callbacks=[cb])

    print(f"\n  Best objective: {study.best_value:.4f} %")
    print(f"  Best Fold-5:    {study.best_trial.user_attrs['fold5_nmae']:.4f} %")
    print(f"  Best Fold-4:    {study.best_trial.user_attrs['fold4_nmae']:.4f} %")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDY_PATH, "wb") as fh:
        pickle.dump(study, fh)

    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auto-generated by tune_era5.py",
        f"# Fold-5 nMAE: {study.best_trial.user_attrs['fold5_nmae']:.4f} %",
        f"# Fold-4 nMAE: {study.best_trial.user_attrs['fold4_nmae']:.4f} %",
        "",
    ]
    for k, v in study.best_params.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.6g}")
        else:
            lines.append(f"{k}: {v}")
    CONFIG_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Params saved to {CONFIG_OUT}")


if __name__ == "__main__":
    main()
