"""V5: Multi-model ensemble (LightGBM + CatBoost + XGBoost) with power curve.

Strategy:
- Train 3 diverse gradient boosting models on the same features.
- Use Fold-5 OOF predictions to find optimal blend weights via ridge regression.
- Generate submission from the weighted ensemble.

This exploits model diversity: each library has different split algorithms,
regularization, and handling of feature interactions.

Usage:
    python -m src.training.train_ensemble
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import Fold, default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.features.power_curve import fit_power_curve
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.5_ensemble.csv"


def _add_power_curve_features(df: pd.DataFrame, pc_lookup) -> pd.DataFrame:
    ws = df["wind_speed_80m"].to_numpy(dtype=float)
    dir_deg = df["wind_direction_80m"].to_numpy(dtype=float) * 1000.0
    n_sectors = pc_lookup.n_sectors
    sector_width = 360.0 / n_sectors
    dir_sector = (dir_deg // sector_width).astype(int) % n_sectors

    df = df.copy()
    df["p_theoretical"] = pc_lookup.predict(ws, dir_sector)
    df["p_theo_ratio"] = df["p_theoretical"] / CAPACITY_MW
    df["p_theo_x_active"] = df["p_theoretical"] * df["active_turbines"] / 26.0

    ws120 = df["wind_speed_120m"].to_numpy(dtype=float)
    df["p_theoretical_120m"] = pc_lookup.predict(ws120, dir_sector)
    df["p_theo_diff_80_120"] = df["p_theoretical"] - df["p_theoretical_120m"]
    return df


def _train_catboost(X_tr, y_tr, X_va, y_va, feat_cols, seed=42):
    """Train CatBoost regressor."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=seed,
        loss_function="MAE",
        eval_metric="MAE",
        early_stopping_rounds=200,
        verbose=0,
        task_type="CPU",
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    return model


def _train_xgboost(X_tr, y_tr, X_va, y_va, feat_cols, seed=42):
    """Train XGBoost regressor."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)

    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": seed,
        "tree_method": "hist",
        "verbosity": 0,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return model


def _predict_catboost(model, X):
    return np.clip(model.predict(X), 0.0, CAPACITY_MW)


def _predict_xgboost(model, X, feat_cols):
    import xgboost as xgb
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    return np.clip(model.predict(dmat), 0.0, CAPACITY_MW)


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    print(f"  Train: {len(df_train)} rows, Valid: {len(df_valid)} rows")

    print("Building features...")
    df_train = build_features(df_train, sort_by_time=False)
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    # LGBM config (tuned).
    lgbm_config = LGBMConfig(
        num_leaves=434,
        min_data_in_leaf=124,
        learning_rate=0.0128,
        feature_fraction=0.644,
        bagging_fraction=0.747,
        bagging_freq=2,
        lambda_l1=0.00736,
        lambda_l2=0.00108,
        num_boost_round=4000,
        early_stopping_rounds=200,
        log_period=0,
    )

    # === Collect OOF predictions from all 3 models on Fold-5 ===
    # Use Folds 1-4 for stacker training, Fold-5 for evaluation.
    folds = default_folds()
    fold5 = folds[-1]

    # For the ensemble, we train on everything up to fold5.train_end,
    # validate on fold5's val window.
    train_idx, val_idx = split_indices(df_train, fold5)

    # Fit power curve on training portion.
    pc = fit_power_curve(df_train.iloc[train_idx])
    df_tr = _add_power_curve_features(df_train.iloc[train_idx], pc)
    df_va = _add_power_curve_features(df_train.iloc[val_idx], pc)
    feat_cols = feature_columns(df_tr)

    X_tr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    y_tr = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
    X_va = df_va[feat_cols].to_numpy(dtype=np.float32)
    y_va = df_va[TARGET_COL].to_numpy(dtype=np.float32)

    print(f"\n  Features: {len(feat_cols)}")
    print(f"  Train: {len(X_tr)}, Val (Fold-5): {len(X_va)}")

    # Train all 3 models.
    print("\n=== Training LightGBM ===")
    lgbm_booster = train_lgbm(X_tr, y_tr, X_va, y_va, feature_names=feat_cols, config=lgbm_config)
    preds_lgbm = predict_lgbm(lgbm_booster, X_va)
    nmae_lgbm = normalized_mae(y_va, preds_lgbm)
    print(f"  LightGBM Fold-5 nMAE: {nmae_lgbm:.4f} %")

    print("\n=== Training CatBoost ===")
    cb_model = _train_catboost(X_tr, y_tr, X_va, y_va, feat_cols)
    preds_cb = _predict_catboost(cb_model, X_va)
    nmae_cb = normalized_mae(y_va, preds_cb)
    print(f"  CatBoost Fold-5 nMAE: {nmae_cb:.4f} %")

    print("\n=== Training XGBoost ===")
    xgb_model = _train_xgboost(X_tr, y_tr, X_va, y_va, feat_cols)
    preds_xgb = _predict_xgboost(xgb_model, X_va, feat_cols)
    nmae_xgb = normalized_mae(y_va, preds_xgb)
    print(f"  XGBoost Fold-5 nMAE: {nmae_xgb:.4f} %")

    # === Simple average ensemble ===
    preds_avg = (preds_lgbm + preds_cb + preds_xgb) / 3.0
    preds_avg = np.clip(preds_avg, 0.0, CAPACITY_MW)
    nmae_avg = normalized_mae(y_va, preds_avg)
    print(f"\n  Simple average ensemble nMAE: {nmae_avg:.4f} %")

    # === Ridge-optimized weights ===
    stack_X = np.column_stack([preds_lgbm, preds_cb, preds_xgb])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(stack_X, y_va)
    preds_ridge = np.clip(ridge.predict(stack_X), 0.0, CAPACITY_MW)
    nmae_ridge = normalized_mae(y_va, preds_ridge)
    print(f"  Ridge stacker nMAE: {nmae_ridge:.4f} %")
    print(f"  Ridge weights: LGBM={ridge.coef_[0]:.3f}, CB={ridge.coef_[1]:.3f}, XGB={ridge.coef_[2]:.3f}, intercept={ridge.intercept_:.3f}")

    # === Choose best approach ===
    best_nmae = min(nmae_avg, nmae_ridge, nmae_lgbm, nmae_cb, nmae_xgb)
    print(f"\n  Best Fold-5 nMAE: {best_nmae:.4f} %")

    # === Full-fit all models and generate submission ===
    print("\n=== Full-fit all models ===")
    pc_full = fit_power_curve(df_train)
    df_train_full = _add_power_curve_features(df_train, pc_full)
    feat_cols = feature_columns(df_train_full)

    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)

    # LightGBM full-fit.
    lgbm_full_config = LGBMConfig(
        **{**lgbm_config.__dict__, "num_boost_round": 2500, "early_stopping_rounds": 9999}
    )
    lgbm_full = train_lgbm(X_full, y_full, feature_names=feat_cols, config=lgbm_full_config)

    # CatBoost full-fit.
    from catboost import CatBoostRegressor
    cb_full = CatBoostRegressor(
        iterations=2500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
        random_seed=42, loss_function="MAE", verbose=0, task_type="CPU",
    )
    cb_full.fit(X_full, y_full, verbose=0)

    # XGBoost full-fit.
    import xgboost as xgb
    dtrain_full = xgb.DMatrix(X_full, label=y_full, feature_names=feat_cols)
    xgb_params = {
        "objective": "reg:absoluteerror", "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "seed": 42, "tree_method": "hist", "verbosity": 0,
    }
    xgb_full = xgb.train(xgb_params, dtrain_full, num_boost_round=2500, verbose_eval=False)

    # === Predict validation set ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_full)
    missing = set(feat_cols) - set(df_valid_pred.columns)
    if missing:
        for col in missing:
            df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)

    p_lgbm = np.clip(lgbm_full.predict(X_valid), 0.0, CAPACITY_MW)
    p_cb = np.clip(cb_full.predict(X_valid), 0.0, CAPACITY_MW)
    dval_xgb = xgb.DMatrix(X_valid, feature_names=feat_cols)
    p_xgb = np.clip(xgb_full.predict(dval_xgb), 0.0, CAPACITY_MW)

    # Use the best blending approach from CV.
    if nmae_ridge <= nmae_avg:
        stack_valid = np.column_stack([p_lgbm, p_cb, p_xgb])
        preds_final = np.clip(ridge.predict(stack_valid), 0.0, CAPACITY_MW)
        print("  Using ridge-stacked ensemble")
    else:
        preds_final = np.clip((p_lgbm + p_cb + p_xgb) / 3.0, 0.0, CAPACITY_MW)
        print("  Using simple average ensemble")

    # Restore original row order.
    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
    preds_ordered = np.empty_like(preds_final)
    preds_ordered[order] = preds_final

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))
    print("\nDone.")


if __name__ == "__main__":
    main()
