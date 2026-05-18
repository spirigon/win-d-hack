"""Compare OOF results across all experiment versions.

Run after training chain completes to see which version performed best.

Usage:
    python src/training/compare_oof_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _ROOT / "data" / "processed"


def _load_oof_nmae(version: str) -> dict[str, float | None]:
    path = PROCESSED_DIR / f"{version}_oof.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)

    results: dict[str, float | None] = {}

    # Check if pred_blend_mw column exists
    if "pred_blend_mw" not in df.columns:
        # v108 uses different column names
        if "pred_mlp_mw" in df.columns:
            pred_col = "pred_mlp_mw"
        else:
            return {}
    else:
        pred_col = "pred_blend_mw"

    for fid in [3, 4, 5]:
        sub = df[df["fold"] == fid]
        if len(sub) == 0:
            results[f"f{fid}"] = None
            continue
        target = sub["target_mw"].to_numpy()
        pred   = sub[pred_col].to_numpy()
        # Normalized MAE
        nmae = float(np.mean(np.abs(target - pred)) / (np.mean(target) / 100.0))
        results[f"f{fid}"] = nmae

    # All folds
    target_all = df["target_mw"].to_numpy()
    pred_all   = df[pred_col].to_numpy()
    results["all"] = float(np.mean(np.abs(target_all - pred_all)) / (np.mean(target_all) / 100.0))
    return results


VERSIONS = [
    "v94", "v97", "v97b", "v98", "v99", "v100", "v101",
    "v102", "v103", "v104", "v105", "v106", "v107", "v108",
    "v109", "v110", "v111", "v112", "v113", "v114",
]


def main() -> None:
    rows = []
    for ver in VERSIONS:
        r = _load_oof_nmae(ver)
        if not r:
            continue
        rows.append({
            "version": ver,
            "F3": r.get("f3"),
            "F4": r.get("f4"),
            "F5": r.get("f5"),
            "All": r.get("all"),
        })

    if not rows:
        print("No OOF parquets found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("F5", na_position="last").reset_index(drop=True)

    print("\n" + "=" * 65)
    print("OOF Results Summary (sorted by Fold-5, lower is better)")
    print("=" * 65)
    print(f"{'Version':<10} {'F3':>10} {'F4':>10} {'F5':>10} {'All':>10}")
    print("-" * 65)
    for _, row in df.iterrows():
        f3  = f"{row['F3']:.4f}%" if row["F3"] is not None else "     N/A"
        f4  = f"{row['F4']:.4f}%" if row["F4"] is not None else "     N/A"
        f5  = f"{row['F5']:.4f}%" if row["F5"] is not None else "     N/A"
        all_ = f"{row['All']:.4f}%" if row["All"] is not None else "     N/A"
        marker = " ← BEST" if df.index[0] == _ else ""
        print(f"{row['version']:<10} {f3:>10} {f4:>10} {f5:>10} {all_:>10}{marker}")

    best = df.iloc[0]
    print("-" * 65)
    print(f"\nBest by F5: {best['version']} ({best['F5']:.4f}%)")
    print(f"Estimated LB (v94 optimism = 0.141pp): {best['F5'] - 0.141:.3f}%")
    print(f"\nTarget: n1 = 7.05%  |  Competition top = 7.2%")
    print(f"Current best LB: v94 = 7.432%")
    print(f"Gap to n1: {best['F5'] - 0.141 - 7.05:.3f}pp")


if __name__ == "__main__":
    main()
