"""Hub-height (85 m) wind speed and consensus shear features.

The turbine hub sits at 85 m, but the primary NWP wind column is at 120 m.
Using 120 m overestimates theoretical power by ~17% on average (v³ scaling,
mean shear ratio v_85/v_120 ≈ 0.944).  The shear correction varies with
atmospheric stability (m), which changes hour-to-hour.

Consensus shear: we compute the power-law exponent from three NWP height
pairs (10→80, 10→120, 80→120) and average them — same philosophy as the
multi-model wind speed consensus (ens3_ws100_mean).  The consensus is more
robust than any single-pair estimate.

Features (lean set — 5 total):
  shear_exp_consensus  : mean of m from (10→80, 10→120, 80→120) height pairs
  wind_speed_85m_pl    : v_120 * (85/120)^m_consensus — hub-height speed
  ds_85m_pl_farm_mw    : theoretical farm power at 85 m hub speed (MW)
  ds_85m_pl_ratio      : ds_85m_pl_farm_mw / CAPACITY_MW
  ds_85m_vs_120m_mw    : ds_85m_pl_farm_mw − ds_120m_farm_mw  (shear correction signal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW
from src.features.datasheet_power_curve import per_turbine_power_kw

HUB_HEIGHT_M = 85.0
_M_MIN = 0.05   # very neutral / offshore
_M_MAX = 0.50   # very stable / strongly stratified


def add_hub_height_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 85 m hub-height features using consensus shear exponent."""
    if "wind_speed_120m" not in df.columns or "wind_speed_10m" not in df.columns:
        return df
    if "air_density" not in df.columns or "active_turbines" not in df.columns:
        return df

    df = df.copy()

    v120 = df["wind_speed_120m"].to_numpy(dtype=np.float64).clip(min=0.1)
    v10  = df["wind_speed_10m"].to_numpy(dtype=np.float64).clip(min=0.1)

    # Shear exponent from full profile (10 → 120 m)
    m_10_120 = (np.log(v120 / v10) / np.log(120.0 / 10.0)).clip(_M_MIN, _M_MAX)

    estimates = [m_10_120]

    # Near-hub shear (80 → 120 m) — highest relevance near 85 m hub
    if "wind_speed_80m" in df.columns:
        v80 = df["wind_speed_80m"].to_numpy(dtype=np.float64).clip(min=0.1)
        m_80_120 = (np.log(v120 / v80) / np.log(120.0 / 80.0)).clip(_M_MIN, _M_MAX)
        estimates.append(m_80_120)
        # Also 10 → 80 m pair
        m_10_80 = (np.log(v80 / v10) / np.log(80.0 / 10.0)).clip(_M_MIN, _M_MAX)
        estimates.append(m_10_80)

    # Consensus: average across all available height-pair estimates
    m = np.mean(estimates, axis=0)
    df["shear_exp_consensus"] = m.astype(np.float32)

    # Power-law extrapolated hub-height (85 m) wind speed
    v_hub = v120 * (HUB_HEIGHT_M / 120.0) ** m
    df["wind_speed_85m_pl"] = v_hub.astype(np.float32)

    # Theoretical farm power at corrected hub speed
    density = df["air_density"].to_numpy(dtype=np.float64)
    active  = df["active_turbines"].to_numpy(dtype=np.float64)
    kw_hub  = per_turbine_power_kw(v_hub, density)
    mw_hub  = np.clip(kw_hub * active / 1000.0, 0.0, CAPACITY_MW)
    df["ds_85m_pl_farm_mw"] = mw_hub.astype(np.float32)
    df["ds_85m_pl_ratio"]   = (mw_hub / CAPACITY_MW).astype(np.float32)

    # Shear correction vs current 120 m theoretical power
    if "ds_120m_farm_mw" in df.columns:
        mw_120 = df["ds_120m_farm_mw"].to_numpy(dtype=np.float64)
        df["ds_85m_vs_120m_mw"] = (mw_hub - mw_120).astype(np.float32)

    return df


def hub_height_columns(df: pd.DataFrame) -> list[str]:
    names = {
        "shear_exp_consensus",
        "wind_speed_85m_pl",
        "ds_85m_pl_farm_mw",
        "ds_85m_pl_ratio",
        "ds_85m_vs_120m_mw",
    }
    return [c for c in df.columns if c in names]
