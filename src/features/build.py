"""High-level feature building that handles train+valid concatenation.

For weather lag/rolling features to be valid at the start of the validation
period, we concatenate train and valid chronologically, compute features on
the combined frame, then split back. This avoids NaN lags at the boundary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_train, load_valid_features
from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.features.pipeline import build_features, feature_columns


def build_train_valid_features(
    train_path: str | Path,
    valid_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load, concatenate, build features, split back.

    Returns
    -------
    df_train : pd.DataFrame
        Training frame with features, sorted chronologically.
    df_valid : pd.DataFrame
        Validation frame with features. Retains ``_submission_row`` for
        restoring original row order at submission time.
    feat_cols : list[str]
        Feature column names to feed the model.
    """
    df_train = load_train(train_path)  # sorted ascending
    df_valid = load_valid_features(valid_path)  # has _submission_row

    # Mark source so we can split back.
    df_train["_source"] = "train"
    df_valid["_source"] = "valid"

    # Concatenate chronologically.
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Build features on the combined frame.
    combined = build_features(combined, sort_by_time=False)  # already sorted

    # Split back.
    train_mask = combined["_source"] == "train"
    valid_mask = combined["_source"] == "valid"

    df_train_out = combined[train_mask].drop(columns=["_source"]).reset_index(drop=True)
    df_valid_out = combined[valid_mask].drop(columns=["_source"]).reset_index(drop=True)

    feat_cols = feature_columns(df_train_out)

    return df_train_out, df_valid_out, feat_cols
