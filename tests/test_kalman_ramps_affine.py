"""Unit tests for the v26 additions: Kalman, ramp features, per-month affine.

Focused on correctness properties, not on Fold-5 lift (which is measured
by the ablation runner ``src.training.train_v26_kalman_ramps``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.ramp import add_ramp_features
from src.postprocess.calibration import PerMonthAffine
from src.postprocess.kalman import NWPBiasKalman, add_kalman_bias_features

# ======================================================================= ramp


def _ramp_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # A piecewise pattern: 10h calm, 10h ramp-up, 10h windy, 10h ramp-down.
    base = np.concatenate(
        [np.full(10, 1.5), np.linspace(1.5, 10.0, 10), np.full(10, 10.0), np.linspace(10.0, 1.0, 10)]
    )
    ws = base + rng.normal(0, 0.1, size=n)
    return pd.DataFrame({"wind_speed_120m": ws})


def test_ramp_features_finite_and_shape():
    df = _ramp_frame()
    out = add_ramp_features(df)
    added = [
        "dv_3h",
        "dv_6h",
        "is_sub_cutin",
        "is_sub_cutin_soft",
        "dv_3h_x_sub_cutin",
        "dv_6h_x_sub_cutin",
        "abs_dv_3h",
        "abs_dv_6h",
    ]
    for col in added:
        assert col in out.columns, col
        assert np.all(np.isfinite(out[col].to_numpy())), col
    # No shape change on input columns.
    assert len(out) == len(df)


def test_ramp_sub_cutin_flags_agree_with_threshold():
    df = pd.DataFrame({"wind_speed_120m": [0.5, 1.5, 2.0, 2.5, 4.0]})
    out = add_ramp_features(df, cutin_threshold=2.0, sub_cutin_bandwidth=1.0)
    # Hard flag: strictly below threshold.
    np.testing.assert_array_equal(out["is_sub_cutin"].to_numpy(), [1, 1, 0, 0, 0])
    # Soft flag: 1.0 at/below 2.0, linear decay to 0.0 at 3.0.
    np.testing.assert_allclose(
        out["is_sub_cutin_soft"].to_numpy(),
        np.array([1.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_ramp_features_no_future_information_leakage():
    """``diff(k)[t]`` must equal ``ws[t] - ws[t-k]``.

    If the implementation accidentally looked ahead, diff values at
    stationary indices near a big jump would differ.
    """
    ws = pd.Series(np.arange(20.0, dtype=float))
    df = pd.DataFrame({"wind_speed_120m": ws})
    out = add_ramp_features(df)
    # For the linear ramp ws[t]=t, dv_3h[t] = 3 for t >= 3, and 0 for t < 3
    # (edge rows are NaN→filled-with-zero).
    expected = np.where(np.arange(20) >= 3, 3.0, 0.0).astype(np.float32)
    np.testing.assert_allclose(out["dv_3h"].to_numpy(), expected, atol=0)


# ====================================================================== kalman


def test_kalman_smoother_matches_constant_bias_on_constant_signal():
    """A constant observed bias should smooth to that constant (after burn-in)."""
    y = np.full(500, 2.5)
    km = NWPBiasKalman().fit(y)
    smoothed = km.smooth(y)
    # After a short burn-in the smoother should track the constant precisely.
    assert np.allclose(smoothed[50:], 2.5, atol=0.01)


def test_kalman_handles_missing_observations_without_propagating_nan():
    y = np.full(200, 1.0)
    # Inject 20 NaNs in the middle. The filter's random-walk state should
    # sail through them without producing NaN in the output.
    y[80:100] = np.nan
    km = NWPBiasKalman().fit(y)
    smoothed = km.smooth(y)
    assert np.all(np.isfinite(smoothed))


def test_kalman_fallback_on_tiny_sample():
    km = NWPBiasKalman(min_obs=100_000)  # intentionally impossible threshold
    km.fit(np.array([0.1, 0.2, 0.3]))
    q, r = km.params
    assert q == km.fallback_q
    assert r == km.fallback_r


def test_add_kalman_bias_features_adds_expected_columns():
    n = 300
    df = pd.DataFrame(
        {
            "wind_speed_120m": np.linspace(1, 10, n) + 1.2,  # NWP high-biased
            "era5_wind_speed_100m": np.linspace(1, 10, n),
        }
    )
    train_mask = np.arange(n) < 200
    out, smoother = add_kalman_bias_features(df, train_mask=train_mask)
    assert "ws120_kalman" in out.columns
    assert "ws120_bias_kalman_smooth" in out.columns
    # After burn-in the smoothed bias should approximate 1.2.
    assert abs(out["ws120_bias_kalman_smooth"].iloc[-1] - 1.2) < 0.15
    # The bias-corrected wind is strictly smaller than the raw NWP wind
    # because the bias is positive.
    assert (out["ws120_kalman"] < df["wind_speed_120m"]).mean() > 0.9


def test_add_kalman_bias_features_requires_both_columns():
    df = pd.DataFrame({"wind_speed_120m": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        add_kalman_bias_features(df, train_mask=np.array([True, True, True]))


# =================================================================== calibration


def test_per_month_affine_recovers_linear_mapping():
    rng = np.random.default_rng(1)
    months = np.tile(np.arange(1, 13).repeat(300), 1)
    # Construct OOF where month m has a known affine: y = (0.9 + m*0.01) * yp + m*0.1
    yp = rng.uniform(0, 90, size=len(months)).astype(float)
    yt = np.empty_like(yp)
    for m in range(1, 13):
        mask = months == m
        yt[mask] = (0.9 + 0.01 * m) * yp[mask] + 0.1 * m
    # Add small noise so polyfit behaves.
    yt = yt + rng.normal(0, 0.2, size=len(yt))

    pma = PerMonthAffine(min_rows=50).fit(yp, yt, months)
    assert len(pma.params) == 12
    for m in range(1, 13):
        slope, intercept = pma.params[m]
        assert abs(slope - (0.9 + 0.01 * m)) < 0.02, m
        assert abs(intercept - 0.1 * m) < 0.2, m


def test_per_month_affine_predict_uses_fallback_for_unseen_month():
    yp = np.array([10.0, 20.0, 30.0])
    months = np.array([1, 1, 1])
    pma = PerMonthAffine(fallback=(1.5, 2.0))
    pma.params = {1: (2.0, 1.0)}
    out = pma.predict(yp, months)
    np.testing.assert_allclose(out, [2.0 * 10 + 1.0, 2.0 * 20 + 1.0, 2.0 * 30 + 1.0])
    # Month 7 not in params → fallback.
    out7 = pma.predict(yp, np.array([7, 7, 7]))
    np.testing.assert_allclose(out7, [1.5 * 10 + 2.0, 1.5 * 20 + 2.0, 1.5 * 30 + 2.0])


def test_per_month_affine_rejects_extreme_slopes():
    """A degenerate OOF (all same y_pred → slope = ±inf) must fall back."""
    yp = np.full(300, 5.0)
    yt = np.arange(300, dtype=float)  # can't be explained by constant yp
    months = np.full(300, 3)
    pma = PerMonthAffine(min_rows=100).fit(yp, yt, months)
    # polyfit returns slope=0 for this degenerate case; 0 < 0.5 so the
    # guard rejects the fit and the parameter dict stays empty.
    assert 3 not in pma.params
