"""Repack v32 submission to single-column format with header.

The organizer expects a CSV with one column (the predicted power output)
plus a header line. Our standard ``write_submission`` writes a 2-column
``(timestamp, target)`` CSV; this script drops the timestamp column from
the existing v32 archive file and writes the single-column equivalent
to ``submissions/final/v32.0_era5v2_mw50_cf50_single.csv``.

Usage:

    python scripts/repack_v32_single_column.py
    python scripts/repack_v32_single_column.py \
        --src submissions/archive/v32.0_era5v2_mw50_cf50.csv \
        --dst submissions/final/v32.0_era5v2_mw50_cf50.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TARGET_COL


DEFAULT_SRC = _ROOT / "submissions" / "archive" / "v32.0_era5v2_mw50_cf50.csv"
DEFAULT_DST = _ROOT / "submissions" / "final"  / "v32.0_era5v2_mw50_cf50.csv"

EXPECTED_ROWS = 2126


def repack(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source submission not found: {src}")

    df = pd.read_csv(src)
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Source CSV is missing the target column ({TARGET_COL!r}). "
            f"Got columns: {list(df.columns)}"
        )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")

    preds = df[TARGET_COL].to_numpy(dtype=np.float64)

    # Format / range guards (same intent as src.inference.submission.validate_submission).
    if np.any(np.isnan(preds)):
        raise ValueError("Predictions contain NaN.")
    if np.any(preds < 0):
        raise ValueError("Predictions contain negative values.")
    if np.any(preds > CAPACITY_MW + 1e-6):
        raise ValueError(f"Predictions exceed capacity ({CAPACITY_MW} MW).")

    out = pd.DataFrame({TARGET_COL: preds})
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    out.to_csv(tmp, index=False, float_format="%.6f", encoding="utf-8")
    tmp.replace(dst)

    # Diagnostics
    print(f"Source : {src}")
    print(f"  rows: {len(df)}, columns: {list(df.columns)}")
    print(f"Target : {dst}")
    print(f"  rows: {len(out)}, columns: {list(out.columns)}")
    print(f"  preds: mean={preds.mean():.3f} MW  std={preds.std():.3f} MW  "
          f"range=[{preds.min():.3f}, {preds.max():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = ap.parse_args()
    repack(args.src, args.dst)


if __name__ == "__main__":
    main()
