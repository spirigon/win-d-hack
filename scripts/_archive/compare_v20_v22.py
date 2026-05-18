"""Compare v20.1 (3-fold CV-bag, LB 7.630) with v22.0 (5-fold CV-bag).

The only difference is including Folds 1-2 in the test prediction average.
Fold 1 (winter 2023, nMAE 8.45%) and Fold 2 (autumn 2023, nMAE 8.30%) are
older data, so they might:
  a) Hurt (if they over-emphasize obsolete patterns)
  b) Help (if they add diversity / smooth predictions further)
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

ARCHIVE = ROOT / "submissions" / "archive"


def load_pred(path):
    df = pd.read_csv(path)
    return df[TIMESTAMP_COL].to_numpy(), df[TARGET_COL].to_numpy()


def main():
    _, p_v20 = load_pred(ARCHIVE / "v20.1_lgbm_cv.csv")
    _, p_v22 = load_pred(ARCHIVE / "v22.0_5fold_avg3.csv")
    _, p_v22_r = load_pred(ARCHIVE / "v22.1_5fold_ridge.csv")
    _, p_v22_s = load_pred(ARCHIVE / "v22.2_5fold_simplex.csv")

    print(f"v20.1 (3-fold):       mean={p_v20.mean():6.3f}  std={p_v20.std():6.3f}  p1={np.quantile(p_v20, 0.01):6.3f}  p99={np.quantile(p_v20, 0.99):6.3f}  zeros={(p_v20<0.5).sum()}")
    print(f"v22.0 (5-fold avg3):  mean={p_v22.mean():6.3f}  std={p_v22.std():6.3f}  p1={np.quantile(p_v22, 0.01):6.3f}  p99={np.quantile(p_v22, 0.99):6.3f}  zeros={(p_v22<0.5).sum()}")
    print(f"v22.1 (5-fold ridge): mean={p_v22_r.mean():6.3f}  std={p_v22_r.std():6.3f}  p1={np.quantile(p_v22_r, 0.01):6.3f}  p99={np.quantile(p_v22_r, 0.99):6.3f}  zeros={(p_v22_r<0.5).sum()}")
    print(f"v22.2 (simplex):      mean={p_v22_s.mean():6.3f}  std={p_v22_s.std():6.3f}  p1={np.quantile(p_v22_s, 0.01):6.3f}  p99={np.quantile(p_v22_s, 0.99):6.3f}  zeros={(p_v22_s<0.5).sum()}")

    # Row diffs.
    d = p_v20 - p_v22
    print(f"\nv20.1 - v22.0 avg3: mean={d.mean():+.3f}  std={d.std():.3f}  abs_mean={np.abs(d).mean():.3f}")
    print(f"  Max disagreement: {np.abs(d).max():.2f} MW")
    print(f"  Corr: {np.corrcoef(p_v20, p_v22)[0, 1]:.6f}")

    # Blend.
    print("\nBlend v20.1 + v22.0 (simple avg):")
    blend = 0.5 * p_v20 + 0.5 * p_v22
    print(f"  mean={blend.mean():.3f}  std={blend.std():.3f}")

    # How close are they to uniform mean 40?
    print("\nDistances to quoted LB mean (40.05):")
    for name, p in [("v20.1", p_v20), ("v22.0", p_v22), ("v22.1 ridge", p_v22_r), ("v22.2 simplex", p_v22_s)]:
        print(f"  {name}: |mean - 40.05| = {abs(p.mean() - 40.05):.3f}")


if __name__ == "__main__":
    main()
