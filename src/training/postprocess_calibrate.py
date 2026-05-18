"""Post-processing: Fold-5 isotonic calibration applied to any submission.

Fits an isotonic (monotone) regression on Fold-5 OOF predictions to correct
systematic bias by prediction level:
  - Low predictions (0-7 m/s): model under-predicts by ~0.6-1.4 MW  → add
  - High predictions (10+ m/s): model over-predicts by ~1.4-2.2 MW  → subtract

Fold-5 is Jan-Mar 2025 — most similar to the test period (Jan-Mar 2026).

Usage:
    # Apply to v75 submission using v75 OOF calibration
    python -m src.training.postprocess_calibrate \\
        --oof  data/processed/v75_oof.parquet \\
        --sub  submissions/archive/v75.0_full_ensemble.csv \\
        --out  submissions/archive/v84.0_isotonic_cal.csv

    # Or via import: calibrate_submission(oof_path, sub_path, out_path)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission


def fit_isotonic(oof_path: Path, fold: int = 5) -> IsotonicRegression:
    """Fit isotonic regression on specified OOF fold (default: Fold-5)."""
    oof = pd.read_parquet(oof_path)
    sub = oof[oof["fold"] == fold].reset_index(drop=True)

    # Determine prediction column (v75 uses pred_blend_mw, others use pred_cf_mw)
    pred_col = "pred_blend_mw" if "pred_blend_mw" in sub.columns else "pred_cf_mw"
    pred = sub[pred_col].to_numpy()
    target = sub["target_mw"].to_numpy()

    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(pred, target)

    # Report improvement on Fold-5
    corrected = ir.predict(pred)
    nmae_raw = float(normalized_mae(target, pred))
    nmae_cal = float(normalized_mae(target, corrected))
    print(f"  OOF Fold-{fold} before calibration: {nmae_raw:.4f}%")
    print(f"  OOF Fold-{fold} after  calibration: {nmae_cal:.4f}%  ({nmae_cal - nmae_raw:+.4f}pp)")

    # Also check all-fold effect (to catch overfitting)
    pred_col_all = "pred_blend_mw" if "pred_blend_mw" in oof.columns else "pred_cf_mw"
    all_pred = oof[pred_col_all].to_numpy()
    all_tgt  = oof["target_mw"].to_numpy()
    all_cal  = ir.predict(all_pred)
    nmae_all_raw = float(normalized_mae(all_tgt, all_pred))
    nmae_all_cal = float(normalized_mae(all_tgt, all_cal))
    print(f"  All-fold before: {nmae_all_raw:.4f}%  after: {nmae_all_cal:.4f}%"
          f"  ({nmae_all_cal - nmae_all_raw:+.4f}pp)")

    return ir


def calibrate_submission(
    oof_path: Path,
    sub_path: Path,
    out_path: Path,
    fold: int = 5,
) -> None:
    print(f"\nFitting Fold-{fold} isotonic calibration on {oof_path.name}...")
    ir = fit_isotonic(oof_path, fold=fold)

    # Load submission CSV — always (col0=datetime, col1=predictions)
    sub = pd.read_csv(sub_path)
    ts_col_name  = sub.columns[0]
    pred_col     = sub.columns[1]
    print(f"\nSubmission: ts='{ts_col_name}'  predictions='{pred_col}'")

    raw_mw = sub[pred_col].to_numpy(dtype=np.float64)
    calibrated_mw = np.clip(ir.predict(raw_mw), 0.0, CAPACITY_MW)

    shift = calibrated_mw - raw_mw
    print(f"  Mean raw: {raw_mw.mean():.3f} MW  ->  calibrated: {calibrated_mw.mean():.3f} MW")
    print(f"  Shift stats: mean={shift.mean():.3f}  std={shift.std():.3f}  "
          f"min={shift.min():.3f}  max={shift.max():.3f} MW")

    # Build reordered output — submission rows are in submission order already
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = sub[ts_col_name].to_numpy()
    # write_submission expects predictions in original submission row order
    write_submission(calibrated_mw, out_path,
                     expected_rows=len(sub), timestamps=timestamps)

    print(f"  Calibrated submission written: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Isotonic post-processing calibration")
    ap.add_argument("--oof", type=Path,
                    default=_ROOT / "data" / "processed" / "v75_oof.parquet")
    ap.add_argument("--sub", type=Path,
                    default=_ROOT / "submissions" / "archive" / "v75.0_full_ensemble.csv")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "submissions" / "archive" / "v84.0_isotonic_cal.csv")
    ap.add_argument("--fold", type=int, default=5, help="OOF fold to fit calibration on (default: 5)")
    args = ap.parse_args()

    print("=" * 72)
    print("Post-processing: Isotonic Calibration")
    print(f"  OOF source  : {args.oof}")
    print(f"  Submission  : {args.sub}")
    print(f"  Output      : {args.out}")
    print(f"  Calibration : Fold-{args.fold} OOF")
    print("=" * 72)

    calibrate_submission(args.oof, args.sub, args.out, fold=args.fold)

    print("\nDone.")


if __name__ == "__main__":
    main()
