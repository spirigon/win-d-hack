"""Build v40 submissions from the saved OOF/test parquets.

The v40 training run wrote both parquets correctly but crashed in
``write_submission`` due to a float32 vs float64 dtype mismatch in the
SubmissionSchema. This script builds two submissions from the saved data:

    submissions/archive/v40.0_structural_cf_only.csv

and reports OOF Fold-5 nMAE for diagnostics.
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
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OOF_PATH = _ROOT / "data" / "processed" / "v40_oof.parquet"
TEST_PATH = _ROOT / "data" / "processed" / "v40_test.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v40.0_structural_cf_only.csv"

TURBINE_RATED_MW = 3.465
FOLD_IDS = [3, 4, 5]


def main():
    oof = pd.read_parquet(OOF_PATH)
    test = pd.read_parquet(TEST_PATH)
    print(f"OOF rows: {len(oof)}  Test rows: {len(test)}")
    print(f"OOF cols: {list(oof.columns)}")

    # Per-fold CF nMAE from OOF.
    print("\nv40 CF-only OOF nMAE per fold:")
    for fid in FOLD_IDS:
        sub = oof[oof["fold"] == fid]
        if "pred_cf_mw" in oof.columns:
            n = float(normalized_mae(sub["target_mw"].to_numpy(),
                                     sub["pred_cf_mw"].to_numpy()))
            print(f"  Fold {fid}: {n:.4f}%")

    # Build CF-only test prediction.
    active = test["active_turbines"].to_numpy(dtype=np.float32)
    cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS if f"test_cf_fold{i}" in test.columns]
    print(f"\nUsing CF columns: {cf_cols}")
    cf_mw_per_fold = np.column_stack([
        np.clip(np.clip(test[c].to_numpy(), 0.0, 1.0) * active * TURBINE_RATED_MW,
                0.0, CAPACITY_MW)
        for c in cf_cols
    ])
    final_mw = cf_mw_per_fold.mean(axis=1).astype(np.float64)  # float64 for SubmissionSchema
    final_mw = np.clip(final_mw, 0.0, CAPACITY_MW)

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
