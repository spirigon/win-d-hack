"""V24: Apply explicit icing derate post-processing to existing submissions.

Based on diagnose_icing_residuals.py (Fold-5 analysis):
- Classic icing (T<0 & precip>0): model predicts +3.77 MW too high
- Rime ice (T<2 & precip>0): model predicts +2.77 MW too high
- Snow + freezing: model predicts +3.98 MW too high
- T<0 alone (no precip): near-zero residual → no derate needed

Implementation:
- Load existing submission
- For rows with T<0 & precip>0: multiply pred by 0.90
- For rows with T<2 & precip>0 (but not already in classic-icing): multiply by 0.92

Apply to v20.1 (LB 7.630) and v22.3 50/50 (LB 7.629).

Usage:
    python -m src.training.apply_icing_derate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ARCHIVE = _ROOT / "submissions" / "archive"

DERATE_CLASSIC_ICING = 0.90  # T<0 & precip>0
DERATE_RIME_ICE = 0.92  # T<2 & precip>0 (weaker since signal is smaller)


def compute_icing_mask(valid_df: pd.DataFrame):
    """Return dict of masks for different icing regimes."""
    t80 = valid_df["temperature_80m"].to_numpy()
    precip = (
        valid_df["rain"].to_numpy()
        + valid_df["showers"].to_numpy()
        + valid_df["snowfall"].to_numpy()
    )
    classic_icing = (t80 < 0) & (precip > 0)
    rime_only = (t80 < 2) & (t80 >= 0) & (precip > 0)  # T in [0, 2) with precip
    return {
        "classic_icing": classic_icing,
        "rime_only": rime_only,
        "total_affected": classic_icing | rime_only,
    }


def apply_derate(preds: np.ndarray, masks: dict, c_factor: float, r_factor: float):
    out = preds.copy()
    out[masks["classic_icing"]] *= c_factor
    out[masks["rime_only"]] *= r_factor
    return np.clip(out, 0, CAPACITY_MW)


def process(src_name: str, valid_df: pd.DataFrame, masks: dict):
    src = ARCHIVE / src_name
    if not src.exists():
        print(f"Skip (not found): {src_name}")
        return
    df = pd.read_csv(src)
    # Ensure order matches valid_df.
    assert (df[TIMESTAMP_COL].values == valid_df[TIMESTAMP_COL].values).all(), \
        f"Timestamp order mismatch for {src_name}"
    preds = df[TARGET_COL].to_numpy()
    ts = df[TIMESTAMP_COL].to_numpy()

    print(f"\n{src_name}:")
    print(f"  Original mean: {preds.mean():.3f}")
    print(f"  Classic icing rows: {masks['classic_icing'].sum()}  "
          f"(mean pred: {preds[masks['classic_icing']].mean():.2f})")
    print(f"  Rime-only rows: {masks['rime_only'].sum()}  "
          f"(mean pred: {preds[masks['rime_only']].mean():.2f})")

    # Apply two derate strengths.
    for strength, (c_fac, r_fac) in [
        ("mild", (0.95, 0.97)),
        ("moderate", (0.90, 0.92)),
        ("strong", (0.85, 0.88)),
    ]:
        new_preds = apply_derate(preds, masks, c_fac, r_fac)
        diff = preds - new_preds
        out_name = src_name.replace(".csv", f"_icing_{strength}.csv")
        out_path = ARCHIVE / out_name
        write_submission(new_preds, out_path, expected_rows=len(valid_df), timestamps=ts)
        affected_change = diff[masks["total_affected"]].mean()
        print(f"  {strength} derate (c={c_fac}, r={r_fac}): "
              f"new mean={new_preds.mean():.3f}, "
              f"affected rows pulled down by {affected_change:.2f} MW avg")


def main():
    valid_df = pd.read_csv(VALID_PATH)
    print(f"Valid set: {len(valid_df)} rows")

    masks = compute_icing_mask(valid_df)
    n_classic = masks["classic_icing"].sum()
    n_rime = masks["rime_only"].sum()
    print(f"  Classic icing (T<0 & precip>0): {n_classic} rows ({n_classic/len(valid_df)*100:.1f}%)")
    print(f"  Rime-only (0<=T<2 & precip>0): {n_rime} rows ({n_rime/len(valid_df)*100:.1f}%)")
    print(f"  Total affected: {masks['total_affected'].sum()} rows")

    # Apply to our best submissions.
    for src in [
        "v20.1_lgbm_cv.csv",           # LB 7.630
        "v22.3_blend_v20_50_50.csv",    # LB 7.629 (current best)
        "v22.0_5fold_avg3.csv",         # LB 7.652
    ]:
        process(src, valid_df, masks)

    print("\nDone. Submissions ready for upload.")


if __name__ == "__main__":
    main()
