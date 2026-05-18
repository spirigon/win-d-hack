"""CSV loaders that normalize timestamps and preserve original row order.

Schema validation (pandera==0.22.1) is wired at this boundary — the single
entry point for raw CSVs into the pipeline (Requirements 4.1–4.7).

``SchemaError`` from ``src.data.schemas`` propagates upward without catch:
fatal violations (duplicate timestamp, over-capacity target, renamed
TARGET_COL) must abort the pipeline immediately.

``ws_180m_is_imputed`` placeholder
-----------------------------------
``load_train`` initialises ``ws_180m_is_imputed = False`` for every row
where ``wind_speed_180m`` is present in the CSV.  The presence check is
performed on the raw parsed frame *before* ``validate_generation`` runs,
because ``GenerationSchema`` uses ``strict="filter"`` which drops columns
not declared in the schema (including ``wind_speed_180m``).  The flag is
then attached to the post-validation frame.  The *real* flag — True
where the Hellmann power-law fill replaced a NaN — is computed by
``add_imputation_flag()`` inside ``src/features/pipeline.py::build_features``
at the ``_impute_180m`` boundary (Requirement 3.1).  The placeholder ensures
the column exists on the training frame from the moment it leaves the loader,
satisfying downstream consumers that check for the column before feature
engineering runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import TIMESTAMP_COL
from src.data.schemas import validate_generation, validate_weather


def _parse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")
    return df


def load_train(path: str | Path) -> pd.DataFrame:
    """Load the training set, sorted chronologically ascending.

    The raw CSV is descending; sorting is necessary so lag/rolling features
    and walk-forward splits behave correctly.

    Calls ``validate_generation`` before returning — ``SchemaError``
    propagates without catch (Requirements 4.1, 4.5).

    A ``ws_180m_is_imputed`` placeholder column (all ``False``) is added
    when ``wind_speed_180m`` is present in the raw CSV.  The presence check
    runs before ``validate_generation`` because ``GenerationSchema`` uses
    ``strict="filter"`` which drops non-schema columns.  The real per-row
    flag is set by ``add_imputation_flag()`` in
    ``src/features/pipeline.py::build_features`` at the Hellmann-fill
    boundary (Requirement 3.1).
    """
    df = _parse(pd.read_csv(path))
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Record whether wind_speed_180m is present BEFORE validate_generation
    # strips non-schema columns (GenerationSchema uses strict="filter").
    _has_180m = "wind_speed_180m" in df.columns

    # Schema validation — SchemaError propagates (Requirements 4.1, 4.5).
    df = validate_generation(df, source_path=path)

    # ws_180m_is_imputed placeholder — set to False here; the real flag is
    # computed by add_imputation_flag() at the Hellmann-fill boundary in
    # src/features/pipeline.py::build_features (Requirement 3.1).
    # We check _has_180m (pre-filter) because validate_generation's
    # strict="filter" drops wind_speed_180m from the returned frame.
    if _has_180m:
        df["ws_180m_is_imputed"] = False

    return df


def load_valid_features(path: str | Path) -> pd.DataFrame:
    """Load the validation feature set.

    IMPORTANT: row order is preserved (descending) because the submission
    CSV must have exactly the same row count and order as the input file.
    A copy of the original order is returned, but we also carry a
    ``_submission_row`` column so feature engineering can safely reorder.

    Calls ``validate_weather`` before returning — ``SchemaError``
    propagates without catch (Requirements 4.1, 4.5).
    """
    df = _parse(pd.read_csv(path))
    df["_submission_row"] = range(len(df))

    # Schema validation — SchemaError propagates (Requirements 4.1, 4.5).
    df = validate_weather(df, source_path=path)

    return df
