"""Isotonic re-calibration of OOF predictions.

The Fold-5 diagnostic on v32 showed a systematic ``+1.32 MW`` bias in the
12-17 m/s band and a small ``+0.27 MW`` over-prediction below cut-in.
Isotonic regression on (predicted_mw, actual_mw) corrects monotone
miscalibration without needing a retrain — pure post-processing.

We fit two separate isotonics (CF and MW legs) on the OOF predictions
across all CV-bag folds, then apply each to the corresponding test
predictions before the 50/50 blend. This way the calibration of each leg
is independent.

Edge handling: the isotonic is fit with ``out_of_bounds='clip'`` so the
test set's [0, 90.09] MW window is fully covered without extrapolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.schema import CAPACITY_MW


def fit_iso_recal(oof_pred: np.ndarray, oof_target: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic ``predicted -> actual`` calibrator.

    The OOF predictions and targets must be aligned 1:1 and contain only
    rows with non-null target. NaN/Inf are dropped defensively.

    Parameters
    ----------
    oof_pred:
        Predicted MW from a CV-bag (averaged across seeds, on the
        validation slice of each fold).
    oof_target:
        Ground-truth MW on the same rows.

    Returns
    -------
    IsotonicRegression
        Fitted calibrator clipped to ``[0, CAPACITY_MW]`` on output.
    """
    p = np.asarray(oof_pred, dtype=float)
    y = np.asarray(oof_target, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y) & (y >= 0)
    if mask.sum() < 100:
        raise ValueError(
            f"Too few valid OOF rows for iso-recal: {int(mask.sum())} (< 100)"
        )
    ir = IsotonicRegression(
        y_min=0.0, y_max=float(CAPACITY_MW),
        out_of_bounds="clip",
        increasing=True,
    )
    ir.fit(p[mask], y[mask])
    return ir


def apply_iso_recal(ir: IsotonicRegression, pred: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator and clip to ``[0, CAPACITY_MW]``."""
    out = ir.predict(np.asarray(pred, dtype=float))
    return np.clip(out, 0.0, CAPACITY_MW)


def evaluate_iso_recal_on_oof(
    oof_df: pd.DataFrame,
    *,
    leg_pred_col: str,
    target_col: str = "target_mw",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Per-fold leave-one-out evaluation: fit on the other folds, score on this fold.

    Returns a small dataframe with ``before_nmae_pct`` and ``after_nmae_pct``
    so the caller can see whether iso-recal helped on every fold or just on
    average.
    """
    from src.eval.metrics import normalized_mae

    rows = []
    for fold_id in sorted(oof_df[fold_col].unique()):
        train_mask = oof_df[fold_col] != fold_id
        eval_mask = oof_df[fold_col] == fold_id
        ir = fit_iso_recal(
            oof_df.loc[train_mask, leg_pred_col].to_numpy(),
            oof_df.loc[train_mask, target_col].to_numpy(),
        )
        before = oof_df.loc[eval_mask, leg_pred_col].to_numpy()
        after = apply_iso_recal(ir, before)
        y = oof_df.loc[eval_mask, target_col].to_numpy()
        rows.append({
            "fold": int(fold_id),
            "n": int(eval_mask.sum()),
            "before_nmae_pct": float(normalized_mae(y, before)),
            "after_nmae_pct": float(normalized_mae(y, after)),
        })
    out = pd.DataFrame(rows)
    out["delta_pp"] = out["after_nmae_pct"] - out["before_nmae_pct"]
    return out
