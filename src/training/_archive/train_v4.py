"""V4: Original tuned features + power curve (no weather lags).

The v3 experiment showed that weather lags hurt Fold-5. The power curve
features are extremely strong (top-3 importance). This version combines
the original feature set with just the power curve lookup.

Usage:
    python -m src.training.train_v4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train, load_valid_features
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.features.power_curve import fit_power_curve
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.4_lgbm_v4.csv"


def _add_power_curve_features(df: pd.DataFrame, pc_lookup) -> pd.DataFrame:
    """Add empirical power curve features."""
    ws = df["wind_speed_80m"].to_numpy(dtype=float)
    dir_deg = df["wind_direction_80m"].to_numpy(dtype=float) * 1000.0
    n_sectors = pc_lookup.n_sectors
    sector_width = 360.0 / n_sectors
    dir_sector = (dir_deg // sector_width).astype(int) % n_sectors

    df = df.copy()
    df["p_theoretical"] = pc_lookup.predict(ws, dir_sector)
    df["p_theo_ratio"] = df["p_theoretical"] / CAPACITY_MW
    df["p_theo_x_active"] = df["p_theoretical"] * df["active_turbines"] / 26.0

    # Also fit at 120 m for diversity.
    ws120 = df["wind_speed_120m"].to_numpy(dtype=float)
    df["p_theoretical_120m"] = pc_lookup.predict(ws120, dir_sector)

    # Residual features: difference between levels' theoretical power.
    df["p_theo_diff_80_120"] = df["p_theoretical"] - df["p_theoretical_120m"]

    return df


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train = load_train(TRAIN_PATH)  # sorted ascending
    df_valid = load_valid_features(VALID_PATH)
    print(f"  Train: {len(df_train)} rows, Valid: {len(df_valid)} rows")

    # Build features WITHOUT lags (no concatenation needed).
    print("Building features (no lags)...")
    df_train = build_features(df_train, sort_by_time=False)  # already sorted
    df_valid_sorted = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid_sorted = build_features(df_valid_sorted, sort_by_time=False)

    # Tuned hyperparameters.
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

    # === Walk-forward CV with power curve ===
    print("\n=== Walk-forward CV (v4: original + power curve, no lags) ===")
    folds = default_folds()
    results: list[float] = []
    best_iters: list[int] = []

    for fold in folds:
        train_idx, val_idx = split_indices(df_train, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        # Fit power curve ONLY on this fold's training data.
        pc = fit_power_curve(df_train.iloc[train_idx])

        df_fold_train = _add_power_curve_features(df_train.iloc[train_idx], pc)
        df_fold_val = _add_power_curve_features(df_train.iloc[val_idx], pc)

        feat_cols = feature_columns(df_fold_train)

        X_tr = df_fold_train[feat_cols].to_numpy(dtype=np.float32)
        y_tr = df_fold_train[TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df_fold_val[feat_cols].to_numpy(dtype=np.float32)
        y_va = df_fold_val[TARGET_COL].to_numpy(dtype=np.float32)

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
    pc_full = fit_power_curve(df_train)
    df_train_full = _add_power_curve_features(df_train, pc_full)
    feat_cols = feature_columns(df_train_full)
    print(f"  {len(feat_cols)} features")

    avg_best_iter = int(np.mean(best_iters))
    full_rounds = max(int(avg_best_iter * 1.2), 1000)
    full_config = LGBMConfig(
        **{**config.__dict__, "num_boost_round": full_rounds, "early_stopping_rounds": 9999, "log_period": 200}
    )
    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    booster = train_lgbm(X_full, y_full, feature_names=feat_cols, config=full_config)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_v4.txt"
    booster.save_model(str(model_path))
    print(f"  Model saved to {model_path} ({booster.num_trees()} trees)")

    # === Predict validation ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_features(df_valid_sorted, pc_full)

    missing = set(feat_cols) - set(df_valid_pred.columns)
    if missing:
        print(f"  WARNING: missing features: {missing}")
        for col in missing:
            df_valid_pred[col] = 0.0

    X_valid = df_valid_pred[feat_cols].to_numpy(dtype=np.float32)
    preds = predict_lgbm(booster, X_valid)

    # Restore original row order.
    order = df_valid_pred["_submission_row"].to_numpy().astype(int)
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
