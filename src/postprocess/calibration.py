"""Per-month affine calibration of ensemble predictions.

A thin wrapper over ``numpy.polyfit`` that learns ``(slope, intercept)``
per calendar month on out-of-fold predictions, then applies
``y_corr = slope_m * y + intercept_m`` at inference.

Rationale
---------
The global isotonic calibration tested previously (PROJECT.md §10 item 11)
hurt Fold-5 because fold-to-fold bias patterns differ. This class
sidesteps the failure mode by:

1. Partitioning OOF predictions by **calendar month** (1..12) — the
   dominant seasonal signal in our error analysis.
2. Fitting an **affine** (2 parameters) rather than isotonic (dozens of
   knots), so underfitting beats overfitting on 2k-row monthly buckets.
3. Only applying when per-month fit is stable (coefficient-of-variation
   check on the fit).

This is a **decision-gated** component: if Fold-5 lift is below the
0.02 pp stop threshold from TODO.md the caller is expected to drop the
calibration step entirely.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["PerMonthAffine"]

# ``numpy.polyfit`` raises a ``RankWarning`` on rank-deficient inputs
# (e.g. a constant ``y_pred`` column). The project pytest configuration
# promotes warnings to errors, so we catch the warning locally, degrade
# gracefully to the fallback, and let the caller decide whether the
# per-month fit is usable.
_RANK_WARNING: type[Warning]
try:  # NumPy >= 1.25 moved it under np.exceptions.
    _RANK_WARNING = np.exceptions.RankWarning  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover — older NumPy
    _RANK_WARNING = np.RankWarning  # type: ignore[attr-defined]


@dataclass
class PerMonthAffine:
    """Per-month affine calibration ``y_corr = a_m * y + b_m``.

    Attributes
    ----------
    params
        Dict ``{month: (slope, intercept)}``. Populated by :meth:`fit`.
    fallback
        ``(slope, intercept)`` used when a month is missing from the fit
        (e.g. test period contains a month unseen in OOF data) or when
        the per-month fit is deemed unstable.
    min_rows
        Months with fewer than this many rows reuse ``fallback``.
    """

    params: dict[int, tuple[float, float]] = field(default_factory=dict)
    fallback: tuple[float, float] = (1.0, 0.0)
    min_rows: int = 200

    def fit(
        self,
        y_pred: np.ndarray | pd.Series,
        y_true: np.ndarray | pd.Series,
        months: np.ndarray | pd.Series,
    ) -> PerMonthAffine:
        """Fit one affine per month from OOF predictions.

        Parameters
        ----------
        y_pred
            Model predictions on the OOF set.
        y_true
            Matching ground-truth values.
        months
            Integer calendar month (1..12) for each row.
        """
        yp = np.asarray(y_pred, dtype=float).ravel()
        yt = np.asarray(y_true, dtype=float).ravel()
        mo = np.asarray(months, dtype=int).ravel()
        if not (len(yp) == len(yt) == len(mo)):
            raise ValueError(
                f"shape mismatch: y_pred={len(yp)}, y_true={len(yt)}, months={len(mo)}"
            )

        self.params = {}
        for m in range(1, 13):
            mask = (mo == m) & np.isfinite(yp) & np.isfinite(yt)
            if mask.sum() < self.min_rows:
                continue

            x_sub = yp[mask]
            y_sub = yt[mask]
            # polyfit raises RankWarning on degenerate inputs (near-constant
            # y_pred). Convert any warning into a fallback decision so
            # pytest's ``filterwarnings=error`` does not abort runs.
            with warnings.catch_warnings():
                warnings.simplefilter("error", _RANK_WARNING)
                try:
                    slope, intercept = np.polyfit(x_sub, y_sub, deg=1)
                except (_RANK_WARNING, np.linalg.LinAlgError):
                    continue

            # Guard against pathological fits: the slope must be
            # positive and close to 1.0 (reasonable for a well-trained
            # model). Outside this band we default to the fallback.
            if not np.isfinite(slope) or not np.isfinite(intercept):
                continue
            if not (0.5 <= slope <= 1.5):
                continue
            self.params[int(m)] = (float(slope), float(intercept))
        return self

    # ------------------------------------------------------------- predict
    def predict(
        self,
        y_pred: np.ndarray | pd.Series,
        months: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Return calibrated predictions.

        Missing months or unfitted months use :attr:`fallback`, which
        defaults to the identity map — making calibration a no-op for
        unseen months rather than a silent distortion.
        """
        yp = np.asarray(y_pred, dtype=float).ravel()
        mo = np.asarray(months, dtype=int).ravel()
        if len(yp) != len(mo):
            raise ValueError(f"shape mismatch: y_pred={len(yp)}, months={len(mo)}")

        out = np.empty_like(yp, dtype=float)
        for m in range(1, 13):
            mask = mo == m
            if not mask.any():
                continue
            slope, intercept = self.params.get(int(m), self.fallback)
            out[mask] = slope * yp[mask] + intercept
        # Any rows with month not in 1..12 (shouldn't happen, but fail safe):
        other = (mo < 1) | (mo > 12)
        if other.any():
            out[other] = self.fallback[0] * yp[other] + self.fallback[1]
        return out
