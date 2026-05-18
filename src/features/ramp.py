"""Ramp-direction features for sub-cut-in / ramp-down rows.

Rationale
---------
PROJECT.md §10 item 12 records that post-hoc cut-in clamping hurt Fold-5.
But the underlying data pattern — 892 hours with ``wind_speed_120m < 2``
yet non-zero power (ramp-down inertia and aggregated effects across 26
turbines) — is real. We let the booster learn this regime by exposing
the *signed 3-h / 6-h wind tendency* plus a soft ``sub_cutin`` indicator
rather than clamping after the fact.

All features respect the project-wide leakage discipline: pandas ``diff``
uses only past values, ``.shift(1)`` semantics are not needed because
``.diff(k)`` at index ``t`` references index ``t - k``, not ``t + 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_ramp_features"]


def add_ramp_features(
    df: pd.DataFrame,
    *,
    wind_col: str = "wind_speed_120m",
    cutin_threshold: float = 2.0,
    sub_cutin_bandwidth: float = 1.0,
) -> pd.DataFrame:
    """Attach ``dv_3h``, ``dv_6h``, ``is_sub_cutin`` and interaction columns.

    Parameters
    ----------
    df
        Chronologically sorted feature frame containing ``wind_col``.
    wind_col
        Name of the primary NWP hub-height wind column.
    cutin_threshold
        Wind speed below which sub-cut-in softening applies (2 m/s matches
        plan.md §8.4).
    sub_cutin_bandwidth
        Width of the soft transition band above the hard cut-in
        threshold. The ``is_sub_cutin_soft`` output is a piecewise-linear
        falloff from 1.0 (at ``wind_col ≤ cutin_threshold``) to 0.0 (at
        ``wind_col ≥ cutin_threshold + sub_cutin_bandwidth``).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the added columns. Guarantees every added
        column is finite (NaN at frame edges is filled with 0.0) — the
        booster treats 0.0 as "no change" which matches the physical
        meaning of a tendency at the start of a series.
    """
    if wind_col not in df.columns:
        raise KeyError(f"add_ramp_features: missing required column {wind_col!r}")

    out = df.copy()
    ws = out[wind_col]

    # Signed tendencies. diff(k)[t] = ws[t] - ws[t - k].
    out["dv_3h"] = ws.diff(3).fillna(0.0).astype(np.float32)
    out["dv_6h"] = ws.diff(6).fillna(0.0).astype(np.float32)

    # Hard sub-cut-in flag — binary, cheap, booster-friendly.
    out["is_sub_cutin"] = (ws < cutin_threshold).astype(np.int8)

    # Soft ramp membership: 1.0 below cut-in, 0.0 above cut-in + bandwidth,
    # linear in between. Captures the 892 hours of non-zero power at
    # wind speeds barely above cut-in.
    soft = (cutin_threshold + sub_cutin_bandwidth - ws) / sub_cutin_bandwidth
    out["is_sub_cutin_soft"] = np.clip(soft, 0.0, 1.0).astype(np.float32)

    # Interactions: ramp-down in sub-cut-in regime is the signature we
    # most want the booster to split on.
    out["dv_3h_x_sub_cutin"] = (out["dv_3h"] * out["is_sub_cutin_soft"]).astype(np.float32)
    out["dv_6h_x_sub_cutin"] = (out["dv_6h"] * out["is_sub_cutin_soft"]).astype(np.float32)

    # Absolute tendency (ramp magnitude, unsigned) — generally useful at
    # the 8-12 m/s partial-load band too.
    out["abs_dv_3h"] = out["dv_3h"].abs().astype(np.float32)
    out["abs_dv_6h"] = out["dv_6h"].abs().astype(np.float32)

    return out
