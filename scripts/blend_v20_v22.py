"""Create a simple 50/50 blend of v20.1 (LB 7.630) and v22.0.

Since both are pure-LGBM CV-bagged models and differ only in number of
folds, blending should yield a slight improvement through variance
reduction.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.schema import TARGET_COL, TIMESTAMP_COL
from src.inference.submission import write_submission

ARCHIVE = ROOT / "submissions" / "archive"
VALID_PATH = ROOT / "data" / "raw" / "valid_features.csv"


def main():
    v20 = pd.read_csv(ARCHIVE / "v20.1_lgbm_cv.csv")
    v22 = pd.read_csv(ARCHIVE / "v22.0_5fold_avg3.csv")
    assert (v20[TIMESTAMP_COL].values == v22[TIMESTAMP_COL].values).all()

    ts = v20[TIMESTAMP_COL].to_numpy()
    p20 = v20[TARGET_COL].to_numpy()
    p22 = v22[TARGET_COL].to_numpy()

    for tag, w in [("60_40", 0.6), ("50_50", 0.5), ("40_60", 0.4)]:
        blend = w * p20 + (1 - w) * p22
        print(f"v20 {w:.0%} + v22 {1-w:.0%}: mean={blend.mean():.3f}  std={blend.std():.3f}")
        out = ARCHIVE / f"v22.3_blend_v20_{tag}.csv"
        write_submission(blend, out, expected_rows=len(ts), timestamps=ts)


if __name__ == "__main__":
    main()
