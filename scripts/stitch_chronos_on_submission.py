"""Apply the Chronos 48-hour stitch on any base submission.

The 2-day Chronos stitch on v34.1 gave LB 7.55 (vs v34.1's 7.56). This
script applies the same Chronos forecast to a different base submission
so we can stack the win onto v61 (refined ERA5 coords) once it finishes.

The Chronos forecast is identical regardless of which LightGBM base
because it's a univariate forecast of ``target_mw`` from history alone.
We re-use the predictions saved in ``submissions/archive/v60.2d_chronos_stitch.csv``
(the first 48 hours of which are pure Chronos median).

Usage:
    python scripts/stitch_chronos_on_submission.py \\
        --base submissions/archive/v61.0_refined_coords.csv \\
        --output submissions/archive/v62.0_v61_plus_chronos48h.csv
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

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
CHRONOS_SUB = _ROOT / "submissions" / "archive" / "v60.2d_chronos_stitch.csv"

STITCH_HOURS = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True,
                    help="base submission to stitch onto")
    ap.add_argument("--output", type=Path, required=True,
                    help="where to write the stitched submission")
    args = ap.parse_args()

    if not args.base.exists():
        raise FileNotFoundError(f"Base submission not found: {args.base}")
    if not CHRONOS_SUB.exists():
        raise FileNotFoundError(f"Chronos source not found: {CHRONOS_SUB}")

    base = pd.read_csv(args.base)
    chronos_src = pd.read_csv(CHRONOS_SUB)
    base[TIMESTAMP_COL] = pd.to_datetime(base[TIMESTAMP_COL])
    chronos_src[TIMESTAMP_COL] = pd.to_datetime(chronos_src[TIMESTAMP_COL])

    base = base.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    chronos_src = chronos_src.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    if not (base[TIMESTAMP_COL] == chronos_src[TIMESTAMP_COL]).all():
        raise ValueError("Base and Chronos submissions have different timestamps")

    # Replace the first 48 hours of base with Chronos's first 48 hours.
    first_q1_idx = int(base[TIMESTAMP_COL].searchsorted(pd.Timestamp("2026-01-01 00:00")))
    end_idx = first_q1_idx + STITCH_HOURS
    print(f"Base rows: {len(base)}  first Q1 idx: {first_q1_idx}")
    print(f"Stitching hours [{first_q1_idx}, {end_idx})")

    final = base[TARGET_COL].to_numpy().astype(np.float64).copy()
    chronos_first48 = chronos_src.loc[first_q1_idx:end_idx - 1, TARGET_COL].to_numpy()
    base_first48 = final[first_q1_idx:end_idx]
    print(f"  base    first 48: mean={base_first48.mean():.2f}, "
          f"range=[{base_first48.min():.2f}, {base_first48.max():.2f}]")
    print(f"  chronos first 48: mean={chronos_first48.mean():.2f}, "
          f"range=[{chronos_first48.min():.2f}, {chronos_first48.max():.2f}]")
    print(f"  MAE delta: {np.abs(base_first48 - chronos_first48).mean():.2f} MW")

    final[first_q1_idx:end_idx] = chronos_first48
    final = np.clip(final, 0.0, CAPACITY_MW)

    # Restore descending order matching valid_features.csv.
    df_valid = load_valid_features(VALID_PATH)
    valid_ts_to_row = {ts: i for i, ts in enumerate(df_valid[TIMESTAMP_COL])}
    n = len(base)
    po = np.empty(n, dtype=np.float64)
    ts_po = np.empty(n, dtype="datetime64[ns]")
    for i, ts in enumerate(base[TIMESTAMP_COL]):
        out_idx = valid_ts_to_row.get(ts)
        if out_idx is None:
            raise ValueError(f"timestamp {ts} not found in valid_features.csv")
        po[out_idx] = final[i]
        ts_po[out_idx] = ts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, args.output, expected_rows=n, timestamps=ts_po)
    print(f"\nSubmission saved: {args.output}")


if __name__ == "__main__":
    main()
