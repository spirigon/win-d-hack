"""Compare v19 and v20 submissions to understand what LGBM CV-bagging fixed.

v19.2 (LGBM full-fit 5 seeds x 2200 rounds): LB 7.678
v20.1 (LGBM CV-bagged 3 folds x 5 seeds): LB 7.630  (-0.048 pp)

The only difference is training methodology. CV-bagging produces more
conservative predictions because early-stopped models see less training data.

What does the v20 LGBM predict differently? Are there systematic patterns
(e.g., less extreme values, lower peak, higher low)?
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
    # Load v19.2 (full-fit LGBM) and v20.1 (CV-bagged LGBM).
    ts_19, p_19 = load_pred(ARCHIVE / "v19.2_lgbm_only.csv")
    ts_20, p_20 = load_pred(ARCHIVE / "v20.1_lgbm_cv.csv")
    ts_21_u, p_21_u = load_pred(ARCHIVE / "v21.0_uniform.csv")
    ts_21_ap, p_21_ap = load_pred(ARCHIVE / "v21.0_avpow4.csv")

    assert (ts_19 == ts_20).all(), "Timestamp mismatch"
    assert (ts_19 == ts_21_u).all()

    print(f"v19.2 full-fit:    mean={p_19.mean():6.3f}  std={p_19.std():6.3f}  p99={np.quantile(p_19, 0.99):6.3f}  zeros={(p_19<0.5).sum()}")
    print(f"v20.1 CV-bagged:   mean={p_20.mean():6.3f}  std={p_20.std():6.3f}  p99={np.quantile(p_20, 0.99):6.3f}  zeros={(p_20<0.5).sum()}")
    print(f"v21.0 uniform:     mean={p_21_u.mean():6.3f}  std={p_21_u.std():6.3f}  p99={np.quantile(p_21_u, 0.99):6.3f}  zeros={(p_21_u<0.5).sum()}")
    print(f"v21.0 av_pow4:     mean={p_21_ap.mean():6.3f}  std={p_21_ap.std():6.3f}  p99={np.quantile(p_21_ap, 0.99):6.3f}  zeros={(p_21_ap<0.5).sum()}")

    diff_19_20 = p_19 - p_20
    print(f"\nv19.2 - v20.1 diff: mean={diff_19_20.mean():+.3f}  std={diff_19_20.std():.3f}  abs_mean={np.abs(diff_19_20).mean():.3f}")
    print(f"  Where p_20 is lower: {(diff_19_20 > 0).sum()} rows")
    print(f"  Where p_20 is higher: {(diff_19_20 < 0).sum()} rows")

    # Compare v20 vs v21 (same ensemble methodology, different weighting).
    diff_20_21 = p_20 - p_21_u
    print(f"\nv20.1 - v21.0_uniform diff: mean={diff_20_21.mean():+.3f}  std={diff_20_21.std():.3f}  abs_mean={np.abs(diff_20_21).mean():.3f}")

    # Look at row-level correlation.
    print(f"\nCorrelations:")
    print(f"  v19.2 vs v20.1:         {np.corrcoef(p_19, p_20)[0, 1]:.6f}")
    print(f"  v20.1 vs v21.0 uniform: {np.corrcoef(p_20, p_21_u)[0, 1]:.6f}")
    print(f"  v20.1 vs v21.0 av_pow4: {np.corrcoef(p_20, p_21_ap)[0, 1]:.6f}")

    # Try blending v19.2 with v20.1.
    print("\nBlend v19.2 + v20.1 (if leaderboard matched Fold-5, this would just average):")
    for w in [0.25, 0.5, 0.75]:
        blend = w * p_19 + (1 - w) * p_20
        print(f"  w_v19={w}: mean={blend.mean():.3f}")


if __name__ == "__main__":
    main()
