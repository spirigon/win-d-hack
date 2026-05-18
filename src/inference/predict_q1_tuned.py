"""Generate Q1-2026 predictions from the tuned LightGBM model."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.features.pipeline import build_features, feature_columns
from src.inference.submission import write_submission
from src.utils.seeding import set_global_seed

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
MODEL_PATH = _ROOT / "models" / "lgbm_tuned.txt"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v0.2_lgbm_tuned.csv"


def main() -> None:
    set_global_seed(42)

    print("Loading validation features...")
    df_valid = load_valid_features(VALID_PATH)
    n_rows = len(df_valid)
    print(f"  {n_rows} rows")

    print("Building features...")
    df_valid = build_features(df_valid)
    _ = feature_columns(df_valid)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model_features = booster.feature_name()

    # Ensure all expected features exist.
    missing = set(model_features) - set(df_valid.columns)
    if missing:
        print(f"  WARNING: missing features from valid set: {missing}")
        for col in missing:
            df_valid[col] = 0.0

    X = df_valid[model_features].to_numpy(dtype=np.float32)
    print("Predicting...")
    preds = np.clip(booster.predict(X), 0.0, 90.09)

    order = df_valid["_submission_row"].to_numpy()
    preds_ordered = np.empty_like(preds)
    preds_ordered[order] = preds

    print("Writing submission...")
    write_submission(preds_ordered, OUTPUT_PATH, expected_rows=n_rows)
    print("Done.")


if __name__ == "__main__":
    main()
