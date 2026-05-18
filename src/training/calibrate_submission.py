"""Post-processing bias calibration for any submission CSV.

Loads an OOF parquet (pred_blend_mw vs target_mw) and fits a wind-speed-binned
additive correction.  Applies the correction to a submission CSV using a
reference wind speed column from the validation features.

Usage:
    python src/training/calibrate_submission.py v99

This loads:
    data/processed/v99_oof.parquet   (OOF predictions + ws_120)
    submissions/archive/v99.0_corr_dedup.csv  (test predictions, row-aligned)

And writes:
    submissions/archive/v99.1_calibrated.csv

The wind-speed-binned calibration uses a leave-fold-out scheme to avoid
using the same fold's OOF predictions to calibrate itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.schema import TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

PROCESSED_DIR   = _ROOT / "data" / "processed"
SUBMISSIONS_DIR = _ROOT / "submissions" / "archive"
VALID_PATH      = _ROOT / "data" / "raw" / "valid_features.csv"

# Map version → submission filename (edit to add new versions)
SUBMISSION_MAP = {
    "v99":  "v99.0_corr_dedup.csv",
    "v100": "v100.0_dir_disagree.csv",
    "v101": "v101.0_catboost.csv",
    "v102": "v102.0_multifold_probe.csv",
    "v103": "v103.0_mae_obj.csv",
    "v104": "v104.0_stack.csv",
    "v105": "v105.0_k120.csv",
    "v97":  "v97.0_dedup_gem.csv",
}

# Wind speed bins for calibration (m/s boundaries)
WS_BINS = [0, 3, 5, 7, 8, 9, 10, 11, 12, 14, 16, 25]


def _bin_index(ws: np.ndarray, bins: list[float]) -> np.ndarray:
    """Return the bin index for each wind speed value."""
    idx = np.digitize(ws, bins) - 1
    return np.clip(idx, 0, len(bins) - 2)


def _compute_calibration(
    ws: np.ndarray,
    pred: np.ndarray,
    actual: np.ndarray,
    bins: list[float],
) -> np.ndarray:
    """Compute per-bin additive correction: correction[bin] = mean(actual-pred) in that bin."""
    n_bins = len(bins) - 1
    corrections = np.zeros(n_bins, dtype=np.float64)
    bin_idx = _bin_index(ws, bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 10:  # need at least 10 samples for reliable estimate
            corrections[b] = float(np.mean(actual[mask] - pred[mask]))
    return corrections


def _apply_calibration(
    ws: np.ndarray,
    pred: np.ndarray,
    corrections: np.ndarray,
    bins: list[float],
    capacity_mw: float = None,
) -> np.ndarray:
    """Apply additive corrections and clip to [0, capacity]."""
    bin_idx = _bin_index(ws, bins)
    corrected = pred + corrections[bin_idx]
    lo = 0.0
    hi = float(pred.max() * 1.5) if capacity_mw is None else capacity_mw
    return np.clip(corrected, lo, hi)


def main(version: str = "v99") -> None:
    oof_path = PROCESSED_DIR / f"{version}_oof.parquet"
    sub_name = SUBMISSION_MAP.get(version)
    if sub_name is None:
        print(f"Unknown version '{version}'. Add it to SUBMISSION_MAP.")
        return
    sub_path = SUBMISSIONS_DIR / sub_name
    out_name = sub_name.replace(".csv", "").rsplit(".", 1)[0] + ".1_calibrated.csv"
    out_path = SUBMISSIONS_DIR / out_name

    if not oof_path.exists():
        print(f"OOF not found: {oof_path}")
        return
    if not sub_path.exists():
        print(f"Submission not found: {sub_path}")
        return

    print("=" * 72)
    print(f"Calibrating {version} using wind-speed-binned additive correction")
    print("=" * 72)

    oof = pd.read_parquet(oof_path)
    folds = sorted(oof["fold"].unique())
    print(f"  OOF rows: {len(oof)} across folds {folds}")

    # Leave-fold-out calibration: compute correction from other folds, apply to this fold
    # This is the unbiased estimate of calibration quality
    oof = oof.copy()
    oof["ws_120"] = oof["ws_120"].to_numpy()
    oof["calib_pred"] = oof["pred_blend_mw"].copy()

    for target_fold in folds:
        val_mask = oof["fold"] == target_fold
        cal_mask = oof["fold"] != target_fold

        ws_cal   = oof.loc[cal_mask, "ws_120"].to_numpy()
        pred_cal = oof.loc[cal_mask, "pred_blend_mw"].to_numpy()
        act_cal  = oof.loc[cal_mask, "target_mw"].to_numpy()
        corrections = _compute_calibration(ws_cal, pred_cal, act_cal, WS_BINS)

        ws_val   = oof.loc[val_mask, "ws_120"].to_numpy()
        pred_val = oof.loc[val_mask, "pred_blend_mw"].to_numpy()
        oof.loc[val_mask, "calib_pred"] = _apply_calibration(ws_val, pred_val, corrections, WS_BINS)

    # Report OOF quality before and after calibration
    nmae_before = float(normalized_mae(oof["target_mw"].to_numpy(), oof["pred_blend_mw"].to_numpy()))
    nmae_after  = float(normalized_mae(oof["target_mw"].to_numpy(), oof["calib_pred"].to_numpy()))
    print(f"\n  OOF nMAE before calibration: {nmae_before:.4f}%")
    print(f"  OOF nMAE after  calibration: {nmae_after:.4f}%")
    print(f"  Improvement: {nmae_before - nmae_after:+.4f}pp")

    for fid in folds:
        sub = oof[oof["fold"] == fid]
        nb = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_blend_mw"].to_numpy()))
        na = float(normalized_mae(sub["target_mw"].to_numpy(), sub["calib_pred"].to_numpy()))
        print(f"    Fold {fid}: {nb:.4f}% -> {na:.4f}%  ({na-nb:+.4f}pp)")

    # Print per-bin corrections (fitted on full OOF, used for test)
    ws_all   = oof["ws_120"].to_numpy()
    pred_all = oof["pred_blend_mw"].to_numpy()
    act_all  = oof["target_mw"].to_numpy()
    test_corrections = _compute_calibration(ws_all, pred_all, act_all, WS_BINS)

    print("\n  Full OOF calibration corrections (for test application):")
    for b in range(len(WS_BINS) - 1):
        lo, hi = WS_BINS[b], WS_BINS[b + 1]
        n = int((_bin_index(ws_all, WS_BINS) == b).sum())
        print(f"    {lo:4.0f}-{hi:4.0f} m/s: {test_corrections[b]:+.3f} MW  (n={n})")

    # Apply test corrections to the submission CSV
    # For test wind speed, load the valid features
    df_valid = load_valid_features(VALID_PATH)
    ws_test = df_valid["wind_speed_120m"].to_numpy()

    df_sub = pd.read_csv(sub_path)
    ts_col = df_sub.columns[0]
    pred_col = df_sub.columns[1]
    test_preds_orig = df_sub[pred_col].to_numpy(dtype=np.float64)

    # valid features and submission are in the same row order (per write_submission)
    # but we need to be careful: df_valid is sorted by timestamp, submission might differ.
    # Use TIMESTAMP matching to be safe.
    sub_ts = pd.to_datetime(df_sub[ts_col])
    valid_ts = pd.to_datetime(df_valid[TIMESTAMP_COL])

    # Build alignment: for each submission row, find its ws_120
    ts_to_ws = dict(zip(valid_ts, ws_test))
    ws_sub = np.array([ts_to_ws.get(ts, np.nan) for ts in sub_ts])
    missing = np.isnan(ws_sub).sum()
    if missing > 0:
        print(f"WARNING: {missing} test rows have no matching ws_120 — using mean")
        ws_sub = np.where(np.isnan(ws_sub), float(np.nanmean(ws_test)), ws_sub)

    test_preds_cal = _apply_calibration(ws_sub, test_preds_orig, test_corrections, WS_BINS)

    print(f"\n  Test: mean before={test_preds_orig.mean():.2f} MW  after={test_preds_cal.mean():.2f} MW")
    print(f"  Net shift: {(test_preds_cal - test_preds_orig).mean():+.3f} MW")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(test_preds_cal, out_path, expected_rows=len(df_sub),
                     timestamps=sub_ts.to_numpy())
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v99"
    main(ver)
