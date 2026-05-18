"""V2 training: enhanced features + tuned LGBM + full pipeline.

Usage:
    python -m src.training.train_v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.build import build_train_valid_features
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.3_lgbm_v2.csv"


def main() -> None:
    set_global_seed(42)

    print("Building features (train + valid concatenated for lag continuity)...")
    df_train, df_valid, feat_cols = build_train_valid_features(TRAIN_PATH, VALID_PATH)
    print(f"  Train: {len(df_train)} rows, Valid: {len(df_valid)} rows")
    print(f"  Features: {len(feat_cols)}")

    # Drop rows with NaN in features (first few hours due to lags).
    nan_mask = df_train[feat_cols].isna().any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  Dropping {n_nan} train rows with NaN features (lag warm-up)")
        df_train = df_train[~nan_mask].reset_index(drop=True)

    # Tuned hyperparameters from Optuna.
    config = LGBMConfig(
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
        log_period=200,
    )

    # === Walk-forward CV ===
    print("\n=== Walk-forward CV (v2 features) ===")
    folds = default_folds()
    results: list[float] = []
    best_iters: list[int] = []

    for fold in folds:
        train_idx, val_idx = split_indices(df_train, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"  {fold.name}: skipped (empty)")
            continue

        X_tr = df_train.iloc[train_idx][feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_train.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df_train.iloc[val_idx][feat_cols].to_numpy(dtype=np.float32)
        y_va = df_train.iloc[val_idx][TARGET_COL].to_numpy(dtype=np.float32)

        booster = train_lgbm(X_tr, y_tr, X_va, y_va, feature_names=feat_cols, config=config)
        preds = predict_lgbm(booster, X_va)
        fold_nmae = normalized_mae(y_va, preds)
        results.append(fold_nmae)
        best_iters.append(booster.best_iteration)
        print(f"  {fold.name}: nMAE = {fold_nmae:.4f} %  (best_iter={booster.best_iteration})")

    mean_nmae = float(np.mean(results))
    print(f"\n  Mean nMAE: {mean_nmae:.4f} %")
    print(f"  Fold-5 (Q1 surrogate): {results[-1]:.4f} %")

    # === Full-fit ===
    print("\n=== Full-fit ===")
    avg_best_iter = int(np.mean(best_iters))
    full_config = LGBMConfig(
        **{**config.__dict__, "num_boost_round": int(avg_best_iter * 1.2), "early_stopping_rounds": 9999, "log_period": 200}
    )
    X_full = df_train[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train[TARGET_COL].to_numpy(dtype=np.float32)
    booster = train_lgbm(X_full, y_full, feature_names=feat_cols, config=full_config)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_v2.txt"
    booster.save_model(str(model_path))
    print(f"  Model saved to {model_path} ({booster.num_trees()} trees)")

    # === Predict validation ===
    print("\n=== Predicting Q1-2026 ===")
    # Fill any NaN in valid features (shouldn't happen since we concatenated).
    valid_nan = df_valid[feat_cols].isna().any(axis=1).sum()
    if valid_nan > 0:
        print(f"  WARNING: {valid_nan} valid rows have NaN features, filling with 0")
        df_valid[feat_cols] = df_valid[feat_cols].fillna(0)

    X_valid = df_valid[feat_cols].to_numpy(dtype=np.float32)
    preds = predict_lgbm(booster, X_valid)

    # Restore original row order.
    order = df_valid["_submission_row"].to_numpy().astype(int)
    preds_ordered = np.empty_like(preds)
    preds_ordered[order] = preds

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=len(df_valid))

    # Feature importance (top 20).
    print("\n=== Top 20 features by importance (gain) ===")
    imp = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, imp), key=lambda x: -x[1])
    for name, score in feat_imp[:20]:
        print(f"  {name:40s} {score:12.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
