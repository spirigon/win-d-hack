"""Train LightGBM with tuned hyperparameters from Optuna.

Reads best params from ``configs/model/lgbm_tuned.yaml``, runs full 5-fold
walk-forward CV for comparison with the baseline, then refits on the entire
training set.

Usage:
    python -m src.training.train_lgbm_tuned
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_train
from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.pipeline import build_features, feature_columns
from src.models.lightgbm_model import LGBMConfig, predict_lgbm, train_lgbm
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
CONFIG_PATH = _ROOT / "configs" / "model" / "lgbm_tuned.yaml"
MODEL_DIR = _ROOT / "models"
OOF_PATH = _ROOT / "data" / "processed" / "oof_lgbm_tuned.parquet"


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader for the scalar key:value format we write."""
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            parsed: int | float | str = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                parsed = val
        out[key] = parsed
    return out


def _cv(df: pd.DataFrame, feat_cols: list[str], config: LGBMConfig) -> float:
    results: list[float] = []
    oof_records: list[dict] = []
    for fold in default_folds():
        train_idx, val_idx = split_indices(df, fold)
        X_tr = df.iloc[train_idx][feat_cols].to_numpy(dtype=np.float32)
        y_tr = df.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float32)
        X_va = df.iloc[val_idx][feat_cols].to_numpy(dtype=np.float32)
        y_va = df.iloc[val_idx][TARGET_COL].to_numpy(dtype=np.float32)

        booster = train_lgbm(X_tr, y_tr, X_va, y_va, feature_names=feat_cols, config=config)
        preds = predict_lgbm(booster, X_va)
        fold_nmae = normalized_mae(y_va, preds)
        results.append(fold_nmae)
        print(
            f"  {fold.name}: nMAE = {fold_nmae:.4f} %  "
            f"(n_train={len(train_idx)}, n_val={len(val_idx)}, best_iter={booster.best_iteration})"
        )
        for i, idx in enumerate(val_idx):
            oof_records.append({
                "fold": fold.name,
                TIMESTAMP_COL: df.iloc[idx][TIMESTAMP_COL],
                "y_true": y_va[i],
                "y_pred": preds[i],
            })

    mean_nmae = float(np.mean(results))
    print(f"\n  Mean nMAE across {len(results)} folds: {mean_nmae:.4f} %")

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof_records).to_parquet(OOF_PATH, index=False)
    print(f"  OOF predictions saved to {OOF_PATH}")
    return mean_nmae


def _full_fit(df: pd.DataFrame, feat_cols: list[str], config: LGBMConfig) -> None:
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    # Use a fixed number of boosting rounds on full-fit; taken from OOF mean best_iter * 1.1.
    full_cfg_kwargs = {**config.__dict__, "num_boost_round": 3000, "early_stopping_rounds": 9999}
    full_cfg = LGBMConfig(**full_cfg_kwargs)
    booster = train_lgbm(X, y, feature_names=feat_cols, config=full_cfg)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "lgbm_tuned.txt"
    booster.save_model(str(path))
    print(f"  Full-fit model saved to {path}")


def main() -> None:
    set_global_seed(42)
    print("Loading training data...")
    df = load_train(TRAIN_PATH)
    print(f"  {len(df)} rows")

    print("Building features...")
    df = build_features(df)
    feat_cols = feature_columns(df)
    print(f"  {len(feat_cols)} features")

    print(f"Loading tuned params from {CONFIG_PATH}...")
    best = _load_yaml(CONFIG_PATH)
    print(f"  {best}")

    config = LGBMConfig(
        num_leaves=int(best["num_leaves"]),
        min_data_in_leaf=int(best["min_data_in_leaf"]),
        learning_rate=float(best["learning_rate"]),
        feature_fraction=float(best["feature_fraction"]),
        bagging_fraction=float(best["bagging_fraction"]),
        bagging_freq=int(best["bagging_freq"]),
        lambda_l1=float(best["lambda_l1"]),
        lambda_l2=float(best["lambda_l2"]),
        num_boost_round=4000,
        early_stopping_rounds=200,
        log_period=0,
    )

    print("\n=== Walk-forward CV (tuned) ===")
    _cv(df, feat_cols, config)

    print("\n=== Full-fit (tuned) ===")
    _full_fit(df, feat_cols, config)


if __name__ == "__main__":
    main()
