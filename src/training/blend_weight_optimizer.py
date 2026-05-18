"""Scan CF+MW blend weights using saved OOF parquet files.

Usage:
    python src/training/blend_weight_optimizer.py v94
    python src/training/blend_weight_optimizer.py v97
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW
from src.eval.metrics import normalized_mae


def scan_blend(oof: pd.DataFrame, folds: list[int] | None = None) -> None:
    if folds:
        oof = oof[oof["fold"].isin(folds)].copy()

    cf_mw = np.clip(oof["pred_cf_mw"].to_numpy(), 0, CAPACITY_MW)
    mw_mw = np.clip(oof["pred_mw_mw"].to_numpy(), 0, CAPACITY_MW)
    target = oof["target_mw"].to_numpy()

    print(f"{'W_MW':>6}  {'nMAE':>8}")
    print("-" * 18)
    best_w, best_nmae = 0.5, 999.0
    for w in np.arange(0.0, 1.01, 0.05):
        blend = (w * mw_mw + (1 - w) * cf_mw).clip(0, CAPACITY_MW)
        nmae = normalized_mae(target, blend)
        marker = " <-- current" if abs(w - 0.5) < 0.001 else ""
        print(f"  {w:.2f}  {nmae:.4f}%{marker}")
        if nmae < best_nmae:
            best_nmae, best_w = nmae, w

    print(f"\nBest: w_mw={best_w:.2f}  nMAE={best_nmae:.4f}%")


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "v94"
    oof_path = _ROOT / "data" / "processed" / f"{version}_oof.parquet"
    if not oof_path.exists():
        print(f"OOF not found: {oof_path}")
        sys.exit(1)

    oof = pd.read_parquet(oof_path)
    print(f"OOF loaded: {oof_path}  ({len(oof)} rows)")
    print(f"Folds present: {sorted(oof['fold'].unique())}")

    print("\n--- All folds ---")
    scan_blend(oof)

    for fid in [3, 4, 5]:
        if fid in oof["fold"].values:
            print(f"\n--- Fold {fid} only ---")
            scan_blend(oof, folds=[fid])

    # Per-fold optimal blends stacked
    print("\n--- Multi-fold optimal blend (weighted by row count) ---")
    weights_per_fold: dict[int, float] = {}
    for fid in sorted(oof["fold"].unique()):
        sub = oof[oof["fold"] == fid]
        cf_mw = np.clip(sub["pred_cf_mw"].to_numpy(), 0, CAPACITY_MW)
        mw_mw = np.clip(sub["pred_mw_mw"].to_numpy(), 0, CAPACITY_MW)
        target = sub["target_mw"].to_numpy()
        best_w, best_nmae = 0.5, 999.0
        for w in np.arange(0.0, 1.01, 0.05):
            blend = (w * mw_mw + (1 - w) * cf_mw).clip(0, CAPACITY_MW)
            nmae = normalized_mae(target, blend)
            if nmae < best_nmae:
                best_nmae, best_w = nmae, w
        weights_per_fold[fid] = best_w
        print(f"  Fold {fid}: best w_mw={best_w:.2f}  nMAE={best_nmae:.4f}%")


if __name__ == "__main__":
    main()
