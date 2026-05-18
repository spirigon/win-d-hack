"""OOF-chain power lag features.

The target has lag-1h autocorrelation of 0.947 — by far the strongest
signal in the dataset. This module builds lag features that are consistent
between training and validation:

    Training rows (no OOF)  → actual past power (target.shift(k))
    Training rows (has OOF) → prior-model OOF predictions (from oof_parquet)
    Validation rows         → prior-model test predictions (from test_parquet)

Design
------
The combined (train+valid) frame is sorted chronologically. We build a
"power_series" column:
    - For valid rows: prior model's test-set CF predictions (converted to MW)
    - For train rows with OOF coverage: prior model's OOF pred_cf_mw
    - For train rows without OOF (early folds): actual target

This three-tier approach eliminates the train-test distribution shift:
both OOF-covered training rows and validation rows see "noisy predicted"
lags rather than "perfect actual" lags.  Early training rows (no OOF)
retain actual power — their lags don't bleed into the test period.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TOTAL_TURBINES

_ROOT = Path(__file__).resolve().parents[2]
LAGS = [1, 3, 6, 12, 24]


def _from_cf(cf: np.ndarray, active: np.ndarray) -> np.ndarray:
    """CF → MW using per-turbine capacity."""
    return cf * active * (CAPACITY_MW / TOTAL_TURBINES)


def build_power_lag_features(
    combined: pd.DataFrame,
    *,
    test_parquet: str | Path | None = None,
    oof_parquet: str | Path | None = None,
    lags: list[int] = LAGS,
    fillna_value: float = 0.0,
) -> pd.DataFrame:
    """Add lag_power_{k}h columns to the combined (train+valid) frame.

    Parameters
    ----------
    combined:
        Chronologically sorted frame with both train and valid rows.
        Must have ``_split`` column ('train' or 'valid'), ``TARGET_COL``
        (NaN for valid rows), and ``TIMESTAMP_COL``.
    test_parquet:
        Path to a previously saved test predictions parquet.  Contains
        per-fold CF predictions for the validation rows — used to fill
        valid lags.  If None or not found, valid lags are filled with
        ``fillna_value``.
    oof_parquet:
        Path to a previously saved OOF predictions parquet.  Contains
        ``ts`` and ``pred_cf_mw`` columns.  When provided, training rows
        whose timestamps appear in the OOF frame have their actual power
        replaced by the OOF prediction — eliminating the train/test
        distribution shift caused by perfect vs. noisy lags.
    lags:
        Lag offsets in hours. Default: [1, 3, 6, 12, 24].
    fillna_value:
        Fill value for rows where the lag is unavailable.

    Returns
    -------
    pd.DataFrame
        Same frame with new ``lag_power_{k}h`` columns.
    """
    combined = combined.copy()

    # --- Build the power series ---
    # Default: actual target for training rows, NaN for valid rows.
    power_series = combined[TARGET_COL].copy().astype(float)

    # 1. Valid rows → prior-model test predictions
    if test_parquet is not None:
        test_path = Path(test_parquet)
        if test_path.exists():
            test_df = pd.read_parquet(test_path)
            test_df[TIMESTAMP_COL] = pd.to_datetime(test_df[TIMESTAMP_COL])

            fold_cf_cols = [c for c in test_df.columns if c.startswith("test_cf_fold")]
            if fold_cf_cols:
                avg_cf = test_df[fold_cf_cols].mean(axis=1).to_numpy()
                active = test_df["active_turbines"].to_numpy()
                pred_mw = np.clip(_from_cf(avg_cf, active), 0.0, CAPACITY_MW)
                ts_to_pred = dict(zip(test_df[TIMESTAMP_COL], pred_mw))

                valid_mask = combined["_split"] == "valid"
                valid_ts = combined.loc[valid_mask, TIMESTAMP_COL]
                mapped = valid_ts.map(ts_to_pred)
                power_series.loc[valid_mask] = mapped.values

    # 2. Training rows with OOF coverage → prior-model OOF predictions.
    #    This eliminates the train-test distribution shift: both OOF-covered
    #    train rows and valid rows now see "predicted" lags, not actual power.
    if oof_parquet is not None:
        oof_path = Path(oof_parquet)
        if oof_path.exists():
            oof_df = pd.read_parquet(oof_path)
            oof_df["ts"] = pd.to_datetime(oof_df["ts"])
            ts_to_oof = dict(zip(oof_df["ts"], oof_df["pred_cf_mw"]))

            train_mask = combined["_split"] == "train"
            train_ts = combined.loc[train_mask, TIMESTAMP_COL]
            oof_mapped = train_ts.map(ts_to_oof)
            has_oof = oof_mapped.notna()
            # Only replace where OOF prediction is available (folds 3/4/5 coverage)
            power_series.loc[has_oof[has_oof].index] = oof_mapped[has_oof].values

    # --- Compute lags ---
    for k in lags:
        col = f"lag_power_{k}h"
        combined[col] = power_series.shift(k).fillna(fillna_value).values

    # --- Rolling stats on the power series (pure lag, no current-row leakage) ---
    # These use past-only power to estimate persistence and volatility.
    combined["power_roll6h_mean"]  = power_series.shift(1).rolling(6,  min_periods=3).mean().fillna(fillna_value).values
    combined["power_roll12h_mean"] = power_series.shift(1).rolling(12, min_periods=6).mean().fillna(fillna_value).values
    combined["power_roll24h_mean"] = power_series.shift(1).rolling(24, min_periods=12).mean().fillna(fillna_value).values
    combined["power_roll6h_std"]   = power_series.shift(1).rolling(6,  min_periods=3).std().fillna(fillna_value).values
    combined["power_roll12h_std"]  = power_series.shift(1).rolling(12, min_periods=6).std().fillna(fillna_value).values

    # Power ramp: is output rising or falling?
    combined["power_ramp_1h"] = (power_series.shift(1) - power_series.shift(2)).fillna(0.0).values
    combined["power_ramp_3h"] = (power_series.shift(1) - power_series.shift(4)).fillna(0.0).values

    return combined


def power_lag_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns added by build_power_lag_features."""
    lag_cols = [f"lag_power_{k}h" for k in LAGS]
    extra = [
        "power_roll6h_mean", "power_roll12h_mean", "power_roll24h_mean",
        "power_roll6h_std", "power_roll12h_std",
        "power_ramp_1h", "power_ramp_3h",
    ]
    return [c for c in lag_cols + extra if c in df.columns]
