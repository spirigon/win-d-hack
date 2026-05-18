"""V39: V32 CF-leg-only submission (no 50/50 blend with MW).

The Fold-5 OOF analysis showed v32's CF leg alone (7.54%) is significantly
better than the 50/50 mw+cf blend (7.60%). The MW leg drags Fold-5 down.
This script writes a submission using only the CF leg from v32's CV-bag.

Output:
    submissions/archive/v39.0_v32_cf_only.csv

Usage:
    python scripts/make_v39_cf_only.py [--source v32 | v34]
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
from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OOF_BASE = _ROOT / "data" / "processed"
TURBINE_RATED_MW = 3.465
FOLD_IDS = [3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["v32", "v34"], default="v32")
    ap.add_argument("--output", type=Path,
                    default=_ROOT / "submissions" / "archive" / "v39.0_v32_cf_only.csv")
    args = ap.parse_args()

    # OOF sanity check
    oof = pd.read_parquet(OOF_BASE / f"{args.source}_oof.parquet")
    print(f"{args.source} OOF Fold-5 CF-only nMAE:")
    f5 = oof[oof["fold"] == 5]
    n_cf = float(normalized_mae(f5["target_mw"].to_numpy(), f5["pred_cf_mw"].to_numpy()))
    n_mw = float(normalized_mae(f5["target_mw"].to_numpy(), f5["pred_mw_mw"].to_numpy()))
    n_blend = float(normalized_mae(f5["target_mw"].to_numpy(), f5["pred_blend_mw"].to_numpy()))
    print(f"  CF only:  {n_cf:.4f}%")
    print(f"  MW only:  {n_mw:.4f}%")
    print(f"  50/50:    {n_blend:.4f}%")

    # Per-fold CF only
    print(f"\n{args.source} per-fold CF-only nMAE:")
    for fid in FOLD_IDS:
        sub = oof[oof["fold"] == fid]
        n = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_cf_mw"].to_numpy()))
        print(f"  Fold {fid}: {n:.4f}%")

    # Build CF-only test prediction
    test = pd.read_parquet(OOF_BASE / f"{args.source}_test.parquet")
    active = test["active_turbines"].to_numpy(dtype=np.float32)
    cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS if f"test_cf_fold{i}" in test.columns]
    cf_mw_per_fold = np.column_stack([
        np.clip(np.clip(test[c].to_numpy(), 0.0, 1.0) * active * TURBINE_RATED_MW,
                0.0, CAPACITY_MW)
        for c in cf_cols
    ])
    final_mw = cf_mw_per_fold.mean(axis=1)
    final_mw = np.clip(final_mw, 0.0, CAPACITY_MW)

    order = test["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = test[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    df_valid = load_valid_features(VALID_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, args.output, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"\n  Submission saved: {args.output}")
    print(f"  Mean: {final_mw.mean():.2f} MW   "
          f"Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
