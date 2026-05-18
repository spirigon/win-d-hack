"""V11: Season-aware sample weighting + clean power curve + multi-seed.

Hypothesis: Summer (fold-3) is noisy (11.7% nMAE). Training with equal
weight on summer rows might pull the model away from winter patterns.

Experiment: train with reduced weight on April-September rows, see if
Q1-surrogate improves.

Usage:
    python -m src.training.train_v11
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

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
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v1.1_season_weighted.csv"

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


def _post_process(preds, v_eff):
    preds = np.asarray(preds, dtype=float)
    preds = np.where(v_eff < V_CUT_IN, 0.0, preds)
    preds = np.where(v_eff > V_CUT_OUT, np.minimum(preds, 5.0), preds)
    return np.clip(preds, 0.0, CAPACITY_MW)


def compute_season_weights(df, summer_weight: float, outlier_mask, winter_weight: float = 1.0) -> np.ndarray:
    """Weight by season: down-weight April-September.

    summer_weight < 1 reduces the influence of summer rows.
    winter_weight > 1 increases the influence of Dec-Feb (closest to Q1).
    """
    months = df[TIMESTAMP_COL].dt.month.to_numpy()
    # Summer (April-September) = months 4-9.
    is_summer = (months >= 4) & (months <= 9)
    # Deep winter Dec-Feb overlaps with Q1 2026 test set most closely.
    is_deep_winter = np.isin(months, [12, 1, 2])
    weights = np.ones(len(df), dtype=np.float32)
    weights[is_summer] = summer_weight
    weights[is_deep_winter] = winter_weight
    # Zero out physically impossible rows.
    weights[outlier_mask] = 0.0
    return weights


def _train_single(X_tr, y_tr, w_tr, X_va, y_va, feat_cols, seed, config_base):
    cfg = LGBMConfig(**{**config_base.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(
        cfg.to_params(),
        dtrain,
        num_boost_round=4000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def _train_full(X, y, w, feat_cols, seed, n_rounds, config_base):
    cfg = LGBMConfig(**{**config_base.__dict__, "seed": seed})
    dtrain = lgb.Dataset(X, label=y, weight=w, feature_name=feat_cols, free_raw_data=False)
    return lgb.train(cfg.to_params(), dtrain, num_boost_round=n_rounds)


def evaluate(summer_weight: float, winter_weight: float = 1.0) -> tuple[float, float]:
    """Run Fold-5 eval with given summer_weight. Returns (single-seed, ensemble) nMAE."""
    set_global_seed(42)
    df_train = load_train(TRAIN_PATH)
    impossible = identify_impossible_rows(df_train).to_numpy()
    df_train = build_features(df_train, sort_by_time=False)

    config_base = LGBMConfig(
        num_leaves=207, min_data_in_leaf=235, learning_rate=0.02221,
        feature_fraction=0.947, bagging_fraction=0.714, bagging_freq=6,
        lambda_l1=0.00192, lambda_l2=2.059,
        num_boost_round=4000, early_stopping_rounds=200, log_period=0,
    )

    folds = default_folds()
    fold5 = folds[-1]
    train_idx, val_idx = split_indices(df_train, fold5)

    fold_train = df_train.iloc[train_idx]
    fold_outliers = impossible[train_idx]
    fit_data = fold_train[~fold_outliers]
    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])

    df_tr = _add_power_curve_features(fold_train, pc_sector, pc_global)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc_sector, pc_global)
    feat_cols = feature_columns(df_tr)

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    w_tr = compute_season_weights(fold_train, summer_weight, fold_outliers, winter_weight)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    v_eff_va = df_va["v_eff"].to_numpy()

    preds_all = []
    for seed in SEEDS:
        booster = _train_single(X_tr, y_tr, w_tr, X_va, y_va, feat_cols, seed, config_base)
        p = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration), 0, CAPACITY_MW)
        p_pp = _post_process(p, v_eff_va)
        preds_all.append(p_pp)

    single = normalized_mae(y_va, preds_all[0])
    ensemble = normalized_mae(y_va, np.mean(preds_all, axis=0))
    return single, ensemble


def main():
    print("=== Winter upweight ablation on Fold-5 (summer_weight=1.0) ===")
    # Try weights > 1 on winter only.
    for ww in [1.0, 1.25, 1.5, 2.0]:
        single, ensemble = evaluate(summer_weight=1.0, winter_weight=ww)
        print(f"  winter_weight={ww:.2f}: single-seed = {single:.4f} %, ensemble = {ensemble:.4f} %")


if __name__ == "__main__":
    main()
