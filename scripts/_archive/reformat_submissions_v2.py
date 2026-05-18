"""Reformat existing (header-less) submissions to the new header-required format.

New format:
- Header row: METEOFORECASTHOUR_OPENM_Datetime,<target>
- 2126 data rows
- 2127 total lines
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ARCHIVE = _ROOT / "submissions" / "archive"

# Submissions to reformat (from v19 and v20).
TO_REFORMAT = [
    "v19.0_deep_blend.csv",
    "v19.1_ftt_only.csv",
    "v19.2_lgbm_only.csv",
    "v20.0_oof_blend.csv",
    "v20.0_conservative.csv",
    "v20.0_lgbm_heavy.csv",
    "v20.1_lgbm_cv.csv",
]


def main():
    # Read valid_features to get timestamp column in original order.
    valid_df = pd.read_csv(VALID_PATH)
    timestamps = valid_df[TIMESTAMP_COL].to_numpy()
    print(f"Valid features: {len(valid_df)} rows")
    print(f"Timestamp sample: {timestamps[0]} ... {timestamps[-1]}")

    for name in TO_REFORMAT:
        src = ARCHIVE / name
        if not src.exists():
            print(f"  Skip (not found): {name}")
            continue
        # Read existing predictions (header-less).
        preds = pd.read_csv(src, header=None)[0].to_numpy()
        if len(preds) != len(valid_df):
            print(f"  Skip (row count mismatch {len(preds)} vs {len(valid_df)}): {name}")
            continue
        # Re-write with new format.
        out_path = ARCHIVE / name
        write_submission(preds, out_path, expected_rows=len(valid_df), timestamps=timestamps)

    # Verify.
    print("\nVerification:")
    for name in TO_REFORMAT:
        f = ARCHIVE / name
        if not f.exists():
            continue
        with open(f, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        print(f"  {name}: {len(lines)} lines  header={lines[0].strip()[:80]}")


if __name__ == "__main__":
    main()
