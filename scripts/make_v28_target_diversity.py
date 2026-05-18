"""V28: Explore target-diversity blends more aggressively.

v27.1 (50% MW + 50% CF) got LB 7.605 — new best!
The key insight: models trained on DIFFERENT TARGETS make structurally
different errors that cancel when averaged.

Try:
1. Different MW/CF ratios (30/70, 40/60, 60/40, 70/30)
2. Add the logit model as a third target variant (despite worse LB alone,
   it might add diversity in the blend)
3. Triple blend: MW + CF + logit

Usage:
    python scripts/make_v28_target_diversity.py
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

ARCHIVE = _ROOT / "submissions" / "archive"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"


def load_pred(name):
    df = pd.read_csv(ARCHIVE / name)
    return df[TARGET_COL].to_numpy(), df[TIMESTAMP_COL].to_numpy()


def main():
    print("=" * 60)
    print("V28: Target-diversity blends")
    print("=" * 60)

    # Load the three target-variant models.
    p_cf, ts = load_pred("v20.1_lgbm_cv.csv")       # CF target, LB 7.630
    p_mw, _ = load_pred("v27.0_raw_mw_cv3.csv")     # MW target, LB unknown
    p_logit, _ = load_pred("v26.0_logit_cv3.csv")    # Logit target, LB 7.720

    n = len(ts)
    print(f"\nLoaded predictions ({n} rows each):")
    print(f"  CF (v20.1):    mean={p_cf.mean():.3f}")
    print(f"  MW (v27.0):    mean={p_mw.mean():.3f}")
    print(f"  Logit (v26.0): mean={p_logit.mean():.3f}")

    # Correlation matrix.
    corr_cf_mw = np.corrcoef(p_cf, p_mw)[0, 1]
    corr_cf_logit = np.corrcoef(p_cf, p_logit)[0, 1]
    corr_mw_logit = np.corrcoef(p_mw, p_logit)[0, 1]
    print(f"\nCorrelations:")
    print(f"  CF vs MW:    {corr_cf_mw:.6f}")
    print(f"  CF vs Logit: {corr_cf_logit:.6f}")
    print(f"  MW vs Logit: {corr_mw_logit:.6f}")

    # Mean absolute difference (diversity measure).
    print(f"\nMean |diff| (diversity):")
    print(f"  CF vs MW:    {np.abs(p_cf - p_mw).mean():.3f} MW")
    print(f"  CF vs Logit: {np.abs(p_cf - p_logit).mean():.3f} MW")
    print(f"  MW vs Logit: {np.abs(p_mw - p_logit).mean():.3f} MW")

    # --- Two-way blends (MW + CF) ---
    print("\n--- MW + CF blends ---")
    for w_mw in [0.30, 0.40, 0.50, 0.60, 0.70]:
        blend = w_mw * p_mw + (1 - w_mw) * p_cf
        name = f"v28.0_mw{int(w_mw*100)}_cf{int((1-w_mw)*100)}.csv"
        write_submission(blend, ARCHIVE / name, expected_rows=n, timestamps=ts)
        print(f"  {name}: mean={blend.mean():.3f}")

    # --- Three-way blends (MW + CF + Logit) ---
    print("\n--- MW + CF + Logit blends ---")
    # Equal thirds.
    blend_eq = (p_mw + p_cf + p_logit) / 3.0
    write_submission(blend_eq, ARCHIVE / "v28.1_equal_thirds.csv", expected_rows=n, timestamps=ts)
    print(f"  equal_thirds: mean={blend_eq.mean():.3f}")

    # Weighted by inverse LB (CF best, MW unknown, logit worst).
    # Assume MW is ~7.65 (between CF 7.63 and logit 7.72).
    # Weight inversely: CF gets most, logit least.
    for w_cf, w_mw, w_logit, tag in [
        (0.45, 0.40, 0.15, "cf45_mw40_logit15"),
        (0.40, 0.45, 0.15, "cf40_mw45_logit15"),
        (0.35, 0.45, 0.20, "cf35_mw45_logit20"),
        (0.40, 0.40, 0.20, "cf40_mw40_logit20"),
        (0.50, 0.35, 0.15, "cf50_mw35_logit15"),
    ]:
        blend = w_cf * p_cf + w_mw * p_mw + w_logit * p_logit
        name = f"v28.2_{tag}.csv"
        write_submission(blend, ARCHIVE / name, expected_rows=n, timestamps=ts)
        print(f"  {tag}: mean={blend.mean():.3f}")

    # --- Also blend v27.1 (already LB 7.605) with logit for extra diversity ---
    p_v27_1, _ = load_pred("v27.1_mw50_cf50.csv")
    for w_logit in [0.10, 0.15, 0.20]:
        blend = (1 - w_logit) * p_v27_1 + w_logit * p_logit
        name = f"v28.3_v271_{int((1-w_logit)*100)}_logit{int(w_logit*100)}.csv"
        write_submission(blend, ARCHIVE / name, expected_rows=n, timestamps=ts)
        print(f"  v27.1 {int((1-w_logit)*100)}% + logit {int(w_logit*100)}%: mean={blend.mean():.3f}")

    print("\nDone. All blends saved.")


if __name__ == "__main__":
    main()
