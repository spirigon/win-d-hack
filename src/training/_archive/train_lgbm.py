"""Walk-forward CV training + full-fit for LightGBM baseline.

Usage:
    python -m src.training.train_lgbm

Outputs:
- Per-fold nMAE printed to stdout.
- Final model saved to ``models/lgbm_baseline.txt``.
- OOF predictions saved to ``data/processed/oof_lgbm.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable when running as script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
MODEL_DIR = _ROOT / "models"
OOF_PATH = _ROOT / "data" / "processed" / "oof_lgbm.parquet"


def run_cv(df: pd.DataFrame, feat_cols: list[str], config: LGBMConfig) -> float:
    """Run walk-forward CV and return mean nMAE across folds."""
    folds = default_folds()
    results: list[float] = []
    oof_records: list[dict] = []

    for fold in folds:
        train_idx, val_idx = split_indices(df, fold)
        if len(train_idx) == 0 or len(val_idx) == 0:
            print(f"  {fold.name}: skipped (empty split)")
            continue

        X_tr = df.iloc[train_idx][feat_cols].to_numpy(dtype=np.float32)
        y_tr = df.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df.iloc[val_idx][feat_cols].to_numpy(dtype=np.float32)
        y_va = df.iloc[val_idx][TARGET_COL].to_numpy(dtype=np.float32)

        booster = train_lgbm(X_tr, y_tr, X_va, y_va, feature_names=feat_cols, config=config)
        preds = predict_lgbm(booster, X_va)

        fold_nmae = normalized_mae(y_va, preds)
        results.append(fold_nmae)
        print(f"  {fold.name}: nMAE = {fold_nmae:.4f} %  (n_train={len(train_idx)}, n_val={len(val_idx)}, best_iter={booster.best_iteration})")

        for i, idx in enumerate(val_idx):
            oof_records.append({
                "fold": fold.name,
                TIMESTAMP_COL: df.iloc[idx][TIMESTAMP_COL],
                "y_true": y_va[i],
                "y_pred": preds[i],
            })

    mean_nmae = float(np.mean(results))
    print(f"\n  Mean nMAE across {len(results)} folds: {mean_nmae:.4f} %")

    # Save OOF
    oof_df = pd.DataFrame(oof_records)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"  OOF predictions saved to {OOF_PATH}")

    return mean_nmae


def train_full(df: pd.DataFrame, feat_cols: list[str], config: LGBMConfig) -> None:
    """Refit on the entire training set and save the model."""
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)

    # No early stopping on full fit — use best_iteration from last CV fold as num_boost_round.
    full_config = LGBMConfig(
        **{**config.__dict__, "num_boost_round": 3000, "early_stopping_rounds": 9999}
    )
    booster = train_lgbm(X, y, feature_names=feat_cols, config=full_config)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_baseline.txt"
    booster.save_model(str(model_path))
    print(f"  Full model saved to {model_path}")


def main() -> None:
    set_global_seed(42)
    print("Loading training data...")
    df = load_train(TRAIN_PATH)
    print(f"  {len(df)} rows, {df[TIMESTAMP_COL].min()} -> {df[TIMESTAMP_COL].max()}")

    print("Building features...")
    df = build_features(df)
    feat_cols = feature_columns(df)
    print(f"  {len(feat_cols)} features")

    config = LGBMConfig()

    print("\n=== Walk-forward CV ===")
    run_cv(df, feat_cols, config)

    print("\n=== Full-fit ===")
    train_full(df, feat_cols, config)


if __name__ == "__main__":
    main()
