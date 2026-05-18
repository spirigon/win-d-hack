"""Local-level Kalman smoother for the NWP↔ERA5 wind-speed bias.

Background
----------
The dataset's NWP hub-height wind (``wind_speed_120m``, a few m/s above the
farm's 80 m hub) drifts slowly vs. the independent ERA5 reanalysis. A static
bias correction (the ``bias_correction.py`` experiment, v10 archive) hurt
Fold-5 by +0.10 pp because the bias patterns vary fold to fold — a fixed
subtraction cannot track that. A **time-varying** 1-D Kalman filter on a
random-walk bias state sidesteps the failure mode.

Model
-----
Observation ``y_t = wind_speed_120m_t − era5_wind_speed_100m_t`` is treated
as the raw bias signal (available on both train and valid because ERA5 is
merged on both). The unobserved state ``b_t`` is the *slow* component:

.. math::

    b_t &= b_{t-1} + w_t,         \\quad w_t \\sim \\mathcal{N}(0, Q) \\\\
    y_t &= b_t + v_t,             \\quad v_t \\sim \\mathcal{N}(0, R)

``Q`` and ``R`` are fit by MLE on the training subset via
``statsmodels.tsa.UnobservedComponents`` (local-level model). The smoother
is then run across the combined chronological frame to produce two new
feature columns:

* ``ws120_bias_kalman_smooth`` — the smoothed state ``b̂_t``.
* ``ws120_kalman``             — the bias-corrected NWP wind
  ``wind_speed_120m − b̂_t``.

These are fed to the booster as features; no post-hoc subtraction is
applied to predictions. That keeps the tested-and-failed static-bias
mode out of the pipeline.

Leakage
-------
The smoother's output at time ``t`` uses all observations (past and
future) — but the observations themselves are NWP and ERA5 wind speeds,
both of which are available at inference time for the full Q1 2026
horizon. No target information enters. Q/R hyperparameters are fit on
training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["NWPBiasKalman", "add_kalman_bias_features"]


@dataclass
class _FitResult:
    q: float
    r: float
    p0: float


class NWPBiasKalman:
    """1-D local-level Kalman smoother on the NWP↔ERA5 bias signal.

    Parameters
    ----------
    min_obs
        Minimum non-NaN observations required to fit MLE. Below this the
        fallback ``fallback_q`` / ``fallback_r`` are used.
    fallback_q, fallback_r
        Defaults used when MLE fit fails or data is too sparse. The
        ratio ``Q/R ≈ 0.05²`` corresponds to a very slow drift relative
        to hourly noise — a conservative prior.
    """

    def __init__(
        self,
        min_obs: int = 500,
        fallback_q: float = 1e-4,
        fallback_r: float = 0.5,
    ) -> None:
        self.min_obs = int(min_obs)
        self.fallback_q = float(fallback_q)
        self.fallback_r = float(fallback_r)
        self._fit: _FitResult | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train_bias: np.ndarray | pd.Series) -> NWPBiasKalman:
        """Fit Q, R via MLE on the *training* bias series.

        ``train_bias`` is ``wind_speed_120m − era5_wind_speed_100m`` on
        training rows only. NaN positions are ignored by statsmodels.
        """
        y = np.asarray(train_bias, dtype=float)
        finite = np.isfinite(y)
        if finite.sum() < self.min_obs:
            self._fit = _FitResult(q=self.fallback_q, r=self.fallback_r, p0=1.0)
            return self

        try:
            # Lazy import: statsmodels is a heavy dep; avoid paying import
            # cost on modules that merely reference the class.
            from statsmodels.tsa.statespace.structural import UnobservedComponents

            # Local-level = random-walk state + Gaussian observation noise.
            model = UnobservedComponents(y, level="local level")
            # MLE. disp=False keeps the training output quiet.
            res = model.fit(disp=False, method="lbfgs", maxiter=200)
            # Parameter order for "local level": [sigma2.irregular, sigma2.level]
            sigma2_irreg = float(res.params[0])
            sigma2_level = float(res.params[1])
            q = max(sigma2_level, 1e-8)
            r = max(sigma2_irreg, 1e-8)
        except Exception:  # noqa: BLE001 — fall back gracefully on any fit failure
            q = self.fallback_q
            r = self.fallback_r

        # Initial state variance: start diffuse enough that the filter
        # converges quickly within a few hours.
        self._fit = _FitResult(q=q, r=r, p0=10.0)
        return self

    @property
    def params(self) -> tuple[float, float]:
        """Return the fitted ``(Q, R)``. Raises if ``fit`` has not been called."""
        if self._fit is None:
            raise RuntimeError("NWPBiasKalman.fit() must be called before params.")
        return self._fit.q, self._fit.r

    # ------------------------------------------------------- smooth / predict
    def smooth(self, bias: np.ndarray | pd.Series) -> np.ndarray:
        """Return the RTS-smoothed bias series.

        ``bias`` is the full combined (train + valid) series. NaN inputs
        are handled as missing observations — the state simply propagates
        the prior through those rows.
        """
        if self._fit is None:
            raise RuntimeError("NWPBiasKalman.fit() must be called before smooth().")
        y = np.asarray(bias, dtype=float)
        return _rts_smoother_1d(y, q=self._fit.q, r=self._fit.r, p0=self._fit.p0)


def _rts_smoother_1d(y: np.ndarray, *, q: float, r: float, p0: float) -> np.ndarray:
    """1-D Rauch-Tung-Striebel smoother for a local-level model.

    Hand-coded rather than going through the full statsmodels API because
    we want to run across (train + valid) with the same fitted parameters
    without refitting. Pure NumPy — deterministic, vectorized-friendly,
    runs in ~ms on ~40k rows.
    """
    n = len(y)
    if n == 0:
        return np.asarray(y, dtype=float)

    # Forward filter.
    x_pred = np.empty(n, dtype=float)
    p_pred = np.empty(n, dtype=float)
    x_filt = np.empty(n, dtype=float)
    p_filt = np.empty(n, dtype=float)

    # Initial state: zero mean, diffuse variance.
    x_prev = 0.0
    p_prev = p0

    for t in range(n):
        # Predict.
        x_pred[t] = x_prev
        p_pred[t] = p_prev + q

        yt = y[t]
        if np.isfinite(yt):
            # Observation update.
            k = p_pred[t] / (p_pred[t] + r)
            x_filt[t] = x_pred[t] + k * (yt - x_pred[t])
            p_filt[t] = (1.0 - k) * p_pred[t]
        else:
            # Missing observation: state propagates without update.
            x_filt[t] = x_pred[t]
            p_filt[t] = p_pred[t]

        x_prev = x_filt[t]
        p_prev = p_filt[t]

    # Backward smoother.
    x_smooth = x_filt.copy()
    for t in range(n - 2, -1, -1):
        if p_pred[t + 1] > 0:
            a = p_filt[t] / p_pred[t + 1]
            x_smooth[t] = x_filt[t] + a * (x_smooth[t + 1] - x_pred[t + 1])

    return x_smooth


def add_kalman_bias_features(
    df: pd.DataFrame,
    *,
    train_mask: np.ndarray | pd.Series,
    nwp_col: str = "wind_speed_120m",
    era5_col: str = "era5_wind_speed_100m",
    smoother: NWPBiasKalman | None = None,
) -> tuple[pd.DataFrame, NWPBiasKalman]:
    """Attach ``ws120_kalman`` and ``ws120_bias_kalman_smooth`` to ``df``.

    ``df`` must already contain ``nwp_col`` and ``era5_col`` and must be
    **sorted chronologically**. ``train_mask`` selects rows used to fit
    Q and R (typically ``df["_split"] == "train"``).

    Parameters
    ----------
    smoother
        Optional pre-fit ``NWPBiasKalman``. When ``None`` a fresh one is
        fit on ``df[train_mask]``. Exposed so callers can reuse a fitted
        smoother across train/valid splits without double-fitting.

    Returns
    -------
    df_out, smoother
        The augmented frame (copy) and the fitted smoother (for
        introspection / caching).
    """
    if nwp_col not in df.columns or era5_col not in df.columns:
        raise KeyError(
            f"add_kalman_bias_features: both {nwp_col!r} and {era5_col!r} "
            "must be present; merge ERA5 before calling."
        )
    out = df.copy()
    mask = np.asarray(train_mask, dtype=bool)
    if len(mask) != len(out):
        raise ValueError("train_mask length mismatch with df.")

    bias = out[nwp_col].to_numpy(dtype=float) - out[era5_col].to_numpy(dtype=float)

    if smoother is None:
        smoother = NWPBiasKalman().fit(bias[mask])

    smoothed = smoother.smooth(bias)
    out["ws120_bias_kalman_smooth"] = smoothed.astype(np.float32)
    out["ws120_kalman"] = (
        out[nwp_col].to_numpy(dtype=float) - smoothed
    ).astype(np.float32)
    return out, smoother
