"""Air density features for wind power forecasting.

Wind power P ∝ 0.5 * rho * A * v^3.  Winter air (Jan-Mar) at this site is
~5.5% denser than the 1.225 kg/m³ standard — meaning test-period turbines
generate ~5.5% more power for the same wind speed.  The model never sees
this unless we give it density-scaled features.

Features added:
  - air_density_norm : rho / 1.225   (1.0 = standard; ~1.055 in winter)
  - v120_density     : ws_120 * (rho/1.225)^(1/3)  — density-effective wind speed
  - v_eff_density    : v_eff * (rho/1.225)^(1/3)   (if v_eff column present)
  - wpd_density      : ws_120^3 * (rho/1.225)       — wind power density corrected
  - gfs_ws100m_density, ecmwf_ws100m_density  (if those cols present)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_R_DRY = 287.058   # J / (kg·K)  — specific gas constant for dry air
_RHO_STD = 1.225   # kg/m³  — ISA standard density

_TEMP_COL     = "temperature_120m"    # Celsius
_PRESS_COL    = "pressure_msl"        # hPa


def _compute_density(df: pd.DataFrame) -> np.ndarray | None:
    """Return air density array (kg/m³), or None if source columns missing."""
    if _TEMP_COL not in df.columns or _PRESS_COL not in df.columns:
        return None
    T_C = df[_TEMP_COL].to_numpy(dtype=np.float64)
    P_hPa = df[_PRESS_COL].to_numpy(dtype=np.float64)
    T_K = T_C + 273.15
    P_Pa = P_hPa * 100.0
    rho = P_Pa / (_R_DRY * T_K)
    return rho.astype(np.float32)


def add_air_density_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add density-corrected features to df (in-place copy)."""
    rho = _compute_density(df)
    if rho is None:
        return df

    df = df.copy()
    rho_norm = (rho / _RHO_STD).astype(np.float32)   # ~1.055 in Jan
    df["air_density_norm"] = rho_norm

    # Density-corrected hub-height wind speed (cube-root scaling)
    if "wind_speed_120m" in df.columns:
        ws = df["wind_speed_120m"].to_numpy(dtype=np.float32)
        df["v120_density"] = (ws * np.cbrt(rho_norm)).astype(np.float32)
        # Wind power density: P ∝ rho * v^3
        df["wpd_density"] = (ws ** 3 * rho_norm).astype(np.float32)

    # v_eff already blends multiple wind-speed signals; density-scale it too
    if "v_eff" in df.columns:
        veff = df["v_eff"].to_numpy(dtype=np.float32)
        df["v_eff_density"] = (veff * np.cbrt(rho_norm)).astype(np.float32)

    # Individual NWP source density-corrected speeds
    for col in ("gfs_ws100m", "ecmwf_ws100m"):
        if col in df.columns:
            ws_col = df[col].to_numpy(dtype=np.float32)
            df[f"{col}_density"] = (ws_col * np.cbrt(rho_norm)).astype(np.float32)

    return df


def air_density_columns(df: pd.DataFrame) -> list[str]:
    """Return air-density feature columns present in df."""
    prefixes = (
        "air_density_norm",
        "v120_density",
        "wpd_density",
        "v_eff_density",
        "gfs_ws100m_density",
        "ecmwf_ws100m_density",
    )
    return [c for c in df.columns if c in prefixes]
