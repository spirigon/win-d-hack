"""Structural feature additions targeting the 7-12 m/s error band.

The Fold-5 diagnostic on v32 attributed 56 % of its blend error to the
7-12 m/s "steep PC slope" regime. The features here directly encode the
quantities that make this band hard:

    1. ``pc_slope_84`` — local sensitivity ∂P/∂v at the current
       ``(wind, density)``. Computed by finite differences on the
       Siemens Gamesa SG 3.4-132 datasheet bilinear table. Largest
       around 7-12 m/s, near zero below cut-in and above rated.

    2. ``ramp_3h_power_proj`` — ``pc_slope × dv_3h``. Converts a wind
       ramp from m/s into MW of power change. The booster can split on
       this directly to learn "power is dropping in the next hour"
       events.

    3. NWP lead features (``nwp_lead{6,12}h``, ``nwp_ramp_6h``,
       ``dp_future_6h``, ``disagree_lead6h``). The full Q1 2026 NWP
       forecast is given upfront, so ``shift(-h)`` reads
       known-at-forecast-time future values — no target leakage.

All three families are additive to v32's feature pool and named with the
``struct_`` prefix for traceability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TIMESTAMP_COL
from src.features.datasheet_power_curve import per_turbine_power_kw

__all__ = ["add_structural_features", "structural_columns"]

_FD_DELTA_V = 0.5    # m/s for finite-difference PC slope
_NWP_LEAD_HOURS = (3, 6, 12, 24)
_NWP_RAMP_HORIZONS = (6, 12)


def _pc_slope_per_turbine_kw_per_ms(ws: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Forward-difference PC slope (kW/m/s) per turbine.

    Uses ``(P(ws + Δv) - P(ws - Δv)) / (2Δv)`` with Δv = 0.5 m/s. Below
    the cut-in (ws < 3) the slope is forced to 0 because the PC table
    starts at 3 m/s and the underlying physics is "no production".
    """
    ws_plus = ws + _FD_DELTA_V
    ws_minus = np.maximum(ws - _FD_DELTA_V, 0.0)
    p_plus = per_turbine_power_kw(ws_plus, rho)
    p_minus = per_turbine_power_kw(ws_minus, rho)
    slope = (p_plus - p_minus) / (2.0 * _FD_DELTA_V)
    # Below cut-in the table's bilinear interpolation extrapolates to ws=3
    # values; force slope to zero there.
    slope = np.where(ws < 3.0, 0.0, slope)
    # Above cut-out the manufacturer cuts power; slope is meaningless.
    slope = np.where(ws > 25.0, 0.0, slope)
    return slope


def add_pc_slope(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-turbine and farm-level PC slope features.

    Columns added:
        ``struct_pc_slope_kw_per_ms`` — per-turbine slope at current row
        ``struct_pc_slope_farm_mw``   — slope × active_turbines / 1000
                                          (MW per m/s of wind change)
    """
    out = df.copy()
    ws = out["wind_speed_120m"].to_numpy(dtype=np.float64)
    rho = out["air_density"].fillna(1.225).to_numpy(dtype=np.float64)
    slope_kw = _pc_slope_per_turbine_kw_per_ms(ws, rho)
    out["struct_pc_slope_kw_per_ms"] = slope_kw.astype(np.float32)
    n_active = out["active_turbines"].to_numpy(dtype=np.float32)
    out["struct_pc_slope_farm_mw"] = (slope_kw * n_active / 1000.0).astype(np.float32)
    return out


def add_ramp_power_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Project wind ramps into power-space using the PC slope.

    Requires ``struct_pc_slope_farm_mw`` (added by :func:`add_pc_slope`)
    and the existing ``dv_3h`` / ``dv_6h`` columns from
    ``src.features.ramp.add_ramp_features`` OR computes them on-the-fly
    if they are missing.

    Columns added:
        ``struct_ramp_3h_power_mw`` — pc_slope_farm_mw × dv_3h
        ``struct_ramp_6h_power_mw`` — pc_slope_farm_mw × dv_6h

    Both signed: negative means power is projected to drop, positive
    means rise. Most useful when the booster sees a 5+ MW projected
    drop coupled with high disagreement features.
    """
    out = df.copy()
    if "struct_pc_slope_farm_mw" not in out.columns:
        raise KeyError(
            "add_ramp_power_projection requires struct_pc_slope_farm_mw "
            "(call add_pc_slope first)"
        )

    if "dv_3h" not in out.columns:
        out["dv_3h"] = out["wind_speed_120m"].diff(3).fillna(0.0).astype(np.float32)
    if "dv_6h" not in out.columns:
        out["dv_6h"] = out["wind_speed_120m"].diff(6).fillna(0.0).astype(np.float32)

    slope = out["struct_pc_slope_farm_mw"].to_numpy(dtype=np.float32)
    dv3 = out["dv_3h"].to_numpy(dtype=np.float32)
    dv6 = out["dv_6h"].to_numpy(dtype=np.float32)
    out["struct_ramp_3h_power_mw"] = (slope * dv3).astype(np.float32)
    out["struct_ramp_6h_power_mw"] = (slope * dv6).astype(np.float32)
    return out


def add_nwp_lead(df: pd.DataFrame) -> pd.DataFrame:
    """Future NWP lead features.

    The full Q1 2026 NWP forecast is given upfront in the hackathon
    files, so ``shift(-h)`` reads values that are known at forecast
    time. This is **not** target leakage and is explicitly endorsed by
    the design notes in ``solution.zip``'s ``src/features/lags.py``.

    Columns added:
        ``struct_nwp_lead{3,6,12,24}h``      — wind_speed_120m at t+h
        ``struct_nwp_ramp_6h``               — lead6h - current
        ``struct_nwp_ramp_12h``              — lead12h - current
        ``struct_dp_future_6h``              — pressure_msl(t+6) - pressure_msl(t)
        ``struct_dp_future_12h``             — same at +12 h
        ``struct_disagree_lead6h``           — wind_hub_disagreement at t+6
                                               (only if column present)

    Caller MUST pass a frame **sorted ascending by ``ts``** with no time
    gaps mixed across train/valid splits (combined frame is fine because
    the resulting features are causal-after-shift relative to the
    chronological row).
    """
    out = df.copy()
    if "wind_speed_120m" not in out.columns:
        raise KeyError("add_nwp_lead requires 'wind_speed_120m'")

    nwp = out["wind_speed_120m"]
    for h in _NWP_LEAD_HOURS:
        out[f"struct_nwp_lead{h}h"] = nwp.shift(-h).astype(np.float32)
    for h in _NWP_RAMP_HORIZONS:
        out[f"struct_nwp_ramp_{h}h"] = (nwp.shift(-h) - nwp).astype(np.float32)

    if "pressure_msl" in out.columns:
        pres = out["pressure_msl"]
        for h in (6, 12):
            out[f"struct_dp_future_{h}h"] = (pres.shift(-h) - pres).astype(np.float32)

    # NaN at the trailing edge (last 24 rows of the combined frame). Fill
    # with 0 so the booster doesn't see fold-dependent NaN patterns.
    new_cols = [c for c in out.columns if c.startswith("struct_nwp_lead")
                or c.startswith("struct_nwp_ramp_")
                or c.startswith("struct_dp_future_")]
    out[new_cols] = out[new_cols].fillna(0.0)
    return out


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all three structural families in dependency order.

    Order:
        1. ramp.add_ramp_features      → dv_3h, dv_6h, is_sub_cutin*, abs_dv*
        2. add_pc_slope                → struct_pc_slope_*
        3. add_ramp_power_projection   → struct_ramp_*_power_mw
        4. add_nwp_lead                → struct_nwp_lead*, struct_nwp_ramp_*,
                                          struct_dp_future_*

    Step 1 only runs if the canonical ``dv_3h`` is missing from ``df`` —
    if the caller already invoked ramp features upstream we avoid the
    duplicate work.
    """
    out = df.copy()
    if "dv_3h" not in out.columns or "is_sub_cutin" not in out.columns:
        from src.features.ramp import add_ramp_features
        out = add_ramp_features(out, wind_col="wind_speed_120m")

    out = add_pc_slope(out)
    out = add_ramp_power_projection(out)
    out = add_nwp_lead(out)
    return out


def structural_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``struct_*`` columns currently in ``df``."""
    return [c for c in df.columns if c.startswith("struct_")]
