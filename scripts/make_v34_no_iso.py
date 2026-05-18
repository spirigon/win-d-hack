"""V34.1: V34 without iso-recal.

The v34 LOO evaluation showed isotonic recalibration HURT all three folds
by +0.04 to +0.96 pp. The likely cause is per-fold distribution shift
(2024 Q2 vs 2024 Q4 vs 2025 Q1) — the calibrator fit on two folds doesn't
match the third.

This script reads ``data/processed/v34_test.parquet`` (per-fold test
predictions saved by train_v34_curtailment) and produces a 50/50 blend
WITHOUT applying iso-recal. Output:

    submissions/archive/v34.1_no_iso.csv

Usage:

    python scripts/make_v34_no_iso.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.inference.submission import write_submission

V34_TEST_PATH = _ROOT / "data" / "processed" / "v34_test.parquet"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v34.1_no_iso.csv"

TURBINE_RATED_MW = 3.465
BLEND_WEIGHT_MW = 0.50
FOLD_IDS = [3, 4, 5]


def main():
    test = pd.read_parquet(V34_TEST_PATH)
    print(f"Loaded {len(test)} rows from {V34_TEST_PATH}")

    # Average per-fold test predictions across the CV-bag.
    cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS]
    mw_cols = [f"test_mw_fold{i}" for i in FOLD_IDS]
    avg_test_cf = test[cf_cols].mean(axis=1).to_numpy()
    avg_test_mw = test[mw_cols].mean(axis=1).to_numpy()
    active = test["active_turbines"].to_numpy(dtype=np.float32)

    # Convert CF leg to MW.
    p_avail = active * TURBINE_RATED_MW
    pred_cf_mw = np.clip(np.clip(avg_test_cf, 0.0, 1.0) * p_avail, 0.0, CAPACITY_MW)
    pred_mw_mw = np.clip(avg_test_mw, 0.0, CAPACITY_MW)

    # 50/50 blend, no iso-recal.
    final_mw = (BLEND_WEIGHT_MW * pred_mw_mw + (1 - BLEND_WEIGHT_MW) * pred_cf_mw)
    final_mw = np.clip(final_mw, 0.0, CAPACITY_MW)

    # Restore original (descending) submission row order.
    order = test["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = test[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    df_valid = load_valid_features(VALID_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)

    print(f"\n  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   "
          f"Std: {final_mw.std():.2f} MW   "
          f"Range: [{final_mw.min():.2f}, {final_mw.max():.2f}]")


if __name__ == "__main__":
    main()
