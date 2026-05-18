"""V3 training: power curve feature + trimmed weather lags + tuned LGBM.

Key improvements over v2:
- Empirical power curve lookup (fit on train, apply to both) — gives the model
  a strong physics-informed baseline in the 5-10 m/s transition zone.
- Trimmed lag/rolling features (only short-term, proven useful).
- Proper train/valid concatenation for lag continuity.

Usage:
    python -m src.training.train_v3
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
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v0.3_lgbm_v3.csv"


def _add_power_curve_feature(
    df: pd.DataFrame,
    pc_lookup,
    ws_col: str = "wind_speed_80m",
    dir_col: str = "wind_direction_80m",
) -> pd.DataFrame:
    """Add the empirical power curve prediction as a feature."""
    ws = df[ws_col].to_numpy(dtype=float)
    dir_deg = df[dir_col].to_numpy(dtype=float) * 1000.0
    n_sectors = pc_lookup.n_sectors
    sector_width = 360.0 / n_sectors
    dir_sector = (dir_deg // sector_width).astype(int) % n_sectors

    df["p_theoretical"] = pc_lookup.predict(ws, dir_sector)
    # Ratio features: how far is the theoretical from capacity?
    df["p_theo_ratio"] = df["p_theoretical"] / CAPACITY_MW
    # Interaction with active turbines.
    df["p_theo_x_active"] = df["p_theoretical"] * df["active_turbines"] / 26.0
    return df


def main() -> None:
    set_global_seed(42)

    print("Loading data...")
    df_train_raw = load_train(TRAIN_PATH)
    df_valid_raw = load_valid_features(VALID_PATH)
    print(f"  Train: {len(df_train_raw)} rows, Valid: {len(df_valid_raw)} rows")

    # Concatenate for lag continuity, build base features.
    print("Building base features (concatenated)...")
    df_train_raw["_source"] = "train"
    df_valid_raw["_source"] = "valid"
    combined = pd.concat([df_train_raw, df_valid_raw], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined = build_features(combined, sort_by_time=False)

    # Split back.
    df_train = combined[combined["_source"] == "train"].drop(columns=["_source"]).reset_index(drop=True)
    df_valid = combined[combined["_source"] == "valid"].drop(columns=["_source"]).reset_index(drop=True)

    # Drop NaN rows from lag warm-up.
    feat_cols_base = feature_columns(df_train)
    nan_mask = df_train[feat_cols_base].isna().any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  Dropping {n_nan} train rows with NaN (lag warm-up)")
        df_train = df_train[~nan_mask].reset_index(drop=True)

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
    print("\n=== Walk-forward CV (v3: power curve + lags) ===")
    folds = default_folds()
    results: list[float] = []
    best_iters: list[int] = []

    for fold in folds:
        train_idx, val_idx = split_indices(df_train, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        # Fit power curve ONLY on this fold's training data.
        pc = fit_power_curve(df_train.iloc[train_idx])

        # Add power curve feature to both train and val for this fold.
        df_fold_train = _add_power_curve_feature(df_train.iloc[train_idx].copy(), pc)
        df_fold_val = _add_power_curve_feature(df_train.iloc[val_idx].copy(), pc)

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

    # === Full-fit with power curve on all training data ===
    print("\n=== Full-fit ===")
    pc_full = fit_power_curve(df_train)
    df_train_full = _add_power_curve_feature(df_train.copy(), pc_full)
    feat_cols = feature_columns(df_train_full)
    print(f"  {len(feat_cols)} features")

    avg_best_iter = int(np.mean(best_iters))
    full_config = LGBMConfig(
        **{**config.__dict__, "num_boost_round": int(avg_best_iter * 1.2), "early_stopping_rounds": 9999, "log_period": 200}
    )
    X_full = df_train_full[feat_cols].to_numpy(dtype=np.float32)
    y_full = df_train_full[TARGET_COL].to_numpy(dtype=np.float32)
    booster = train_lgbm(X_full, y_full, feature_names=feat_cols, config=full_config)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_v3.txt"
    booster.save_model(str(model_path))
    print(f"  Model saved to {model_path} ({booster.num_trees()} trees)")

    # === Predict validation ===
    print("\n=== Predicting Q1-2026 ===")
    df_valid_pred = _add_power_curve_feature(df_valid.copy(), pc_full)

    # Ensure feature alignment.
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
