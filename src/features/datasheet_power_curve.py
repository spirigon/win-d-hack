"""Manufacturer power curve for Siemens Gamesa SG 3.4-132 / G132-3.465 MW.

Source: datasheet.md — 26 turbines installed × 3.465 MW rated = 90.09 MW farm.

The curve gives electrical power (kW) per turbine as a function of
wind speed at hub height (m/s) and air density (kg/m^3).

Validity:
- Wind shear ≤ 0.3 (10-min average)
- Turbulence intensity: limited range per bin
- Operating temp: -20°C to +30°C

At inference, we bilinearly interpolate (ws, density) to get per-turbine
power, then multiply by active turbine count for farm-level theoretical power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW, TOTAL_TURBINES

# Wind speed grid (m/s).
WS_GRID = np.arange(3, 26, dtype=float)  # 3..25 inclusive

# Air density grid (kg/m^3). Standard 1.225 + range 1.06..1.27.
DENSITY_GRID = np.array([1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.225, 1.24, 1.27])

# Power table: rows = wind speeds (3..25), cols = densities (in DENSITY_GRID order).
# Reading from the datasheet, columns in the source are [1.225, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27].
# We reorder to monotonic density.
_RAW_TABLE_COL_ORDER = [1.225, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27]
_RAW_POWER_KW = np.array([
    # ws=3
    [37,   29,   30,   32,   33,   35,   36,   38,   39],
    # ws=4
    [169,  139,  144,  150,  155,  161,  167,  172,  178],
    # ws=5
    [434,  363,  376,  389,  402,  415,  428,  441,  454],
    # ws=6
    [816,  697,  719,  740,  762,  784,  805,  826,  848],
    # ws=7
    [1327, 1142, 1176, 1209, 1243, 1277, 1311, 1344, 1378],
    # ws=8
    [1994, 1724, 1774, 1823, 1873, 1922, 1970, 2019, 2067],
    # ws=9
    [2718, 2410, 2471, 2530, 2586, 2641, 2693, 2742, 2789],
    # ws=10
    [3208, 3004, 3050, 3092, 3129, 3164, 3194, 3222, 3247],
    # ws=11
    [3402, 3321, 3341, 3359, 3373, 3386, 3397, 3406, 3414],
    # ws=12
    [3452, 3430, 3436, 3441, 3445, 3448, 3451, 3453, 3455],
    # ws=13
    [3463, 3458, 3459, 3460, 3461, 3462, 3462, 3463, 3463],
    # ws=14
    [3465, 3464, 3464, 3464, 3464, 3464, 3465, 3465, 3465],
    # ws=15
    [3465, 3465, 3465, 3465, 3465, 3465, 3465, 3465, 3465],
    # ws=16
    [3465, 3465, 3465, 3465, 3465, 3465, 3465, 3465, 3465],
    # ws=17
    [3463, 3463, 3463, 3463, 3463, 3463, 3463, 3463, 3463],
    # ws=18
    [3452, 3452, 3452, 3452, 3452, 3452, 3452, 3452, 3452],
    # ws=19
    [3413, 3413, 3413, 3413, 3413, 3413, 3413, 3413, 3413],
    # ws=20
    [3325, 3325, 3325, 3325, 3325, 3325, 3325, 3325, 3325],
    # ws=21
    [3176, 3176, 3176, 3176, 3176, 3176, 3176, 3176, 3176],
    # ws=22
    [2982, 2982, 2982, 2982, 2982, 2982, 2982, 2982, 2982],
    # ws=23
    [2771, 2771, 2771, 2771, 2771, 2771, 2771, 2771, 2771],
    # ws=24
    [2576, 2576, 2576, 2576, 2576, 2576, 2576, 2576, 2576],
    # ws=25
    [2418, 2418, 2418, 2418, 2418, 2418, 2418, 2418, 2418],
], dtype=float)


def _build_monotonic_table() -> np.ndarray:
    """Reorder columns to ascending density. Insert 1.225 in correct position."""
    # Target order: ascending density.
    target_densities = np.sort(DENSITY_GRID)
    # Find each target's position in _RAW_TABLE_COL_ORDER.
    reordered = np.zeros_like(_RAW_POWER_KW)
    for new_idx, d in enumerate(target_densities):
        old_idx = _RAW_TABLE_COL_ORDER.index(d)
        reordered[:, new_idx] = _RAW_POWER_KW[:, old_idx]
    return reordered, target_densities


POWER_TABLE_KW, DENSITY_GRID_SORTED = _build_monotonic_table()


def per_turbine_power_kw(
    wind_speed_ms: np.ndarray,
    air_density: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of manufacturer power curve.

    Returns per-turbine electrical power in kW.
    """
    ws = np.asarray(wind_speed_ms, dtype=float)
    rho = np.asarray(air_density, dtype=float)

    # Clip to table bounds.
    ws_c = np.clip(ws, WS_GRID[0], WS_GRID[-1])
    rho_c = np.clip(rho, DENSITY_GRID_SORTED[0], DENSITY_GRID_SORTED[-1])

    # Locate bin indices.
    ws_lo_idx = np.searchsorted(WS_GRID, ws_c, side="right") - 1
    ws_lo_idx = np.clip(ws_lo_idx, 0, len(WS_GRID) - 2)
    rho_lo_idx = np.searchsorted(DENSITY_GRID_SORTED, rho_c, side="right") - 1
    rho_lo_idx = np.clip(rho_lo_idx, 0, len(DENSITY_GRID_SORTED) - 2)

    # Fractions.
    ws_hi_idx = ws_lo_idx + 1
    rho_hi_idx = rho_lo_idx + 1

    ws_frac = (ws_c - WS_GRID[ws_lo_idx]) / (WS_GRID[ws_hi_idx] - WS_GRID[ws_lo_idx])
    rho_frac = (rho_c - DENSITY_GRID_SORTED[rho_lo_idx]) / (DENSITY_GRID_SORTED[rho_hi_idx] - DENSITY_GRID_SORTED[rho_lo_idx])

    # Bilinear values.
    ll = POWER_TABLE_KW[ws_lo_idx, rho_lo_idx]
    lh = POWER_TABLE_KW[ws_lo_idx, rho_hi_idx]
    hl = POWER_TABLE_KW[ws_hi_idx, rho_lo_idx]
    hh = POWER_TABLE_KW[ws_hi_idx, rho_hi_idx]

    # Linear in both dimensions.
    interp = (
        ll * (1 - ws_frac) * (1 - rho_frac)
        + lh * (1 - ws_frac) * rho_frac
        + hl * ws_frac * (1 - rho_frac)
        + hh * ws_frac * rho_frac
    )

    # Below cut-in (3 m/s) → 0.
    interp = np.where(ws < 3.0, 0.0, interp)
    # Above cut-out (25 m/s) → keep last value (storm mode). We could also
    # set to 0 here but the turbine still produces until ~25 m/s.
    # Above 27 m/s we assume fully shut.
    interp = np.where(ws > 27.0, 0.0, interp)

    return interp


def farm_theoretical_power_mw(
    wind_speed_ms: np.ndarray,
    air_density: np.ndarray,
    active_turbines: np.ndarray | int = TOTAL_TURBINES,
) -> np.ndarray:
    """Theoretical farm-level power (MW) given wind speed, density, and active turbine count."""
    per_turbine_kw = per_turbine_power_kw(wind_speed_ms, air_density)
    if np.isscalar(active_turbines):
        active_turbines = np.full_like(per_turbine_kw, active_turbines, dtype=float)
    else:
        active_turbines = np.asarray(active_turbines, dtype=float)
    farm_kw = per_turbine_kw * active_turbines
    return np.clip(farm_kw / 1000.0, 0.0, CAPACITY_MW)


def add_datasheet_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add manufacturer power curve features to a dataframe.

    Requires columns: wind_speed_80m, wind_speed_120m, air_density,
    active_turbines.
    """
    df = df.copy()
    active = df["active_turbines"].to_numpy()
    density = df["air_density"].to_numpy()

    # Per-turbine theoretical power at different hub-height proxies.
    for ws_col, label in (
        ("wind_speed_80m", "ds_80m"),
        ("wind_speed_120m", "ds_120m"),
        ("era5_wind_speed_100m", "ds_era5_100m"),
    ):
        if ws_col in df.columns:
            ws_arr = df[ws_col].to_numpy()
            per_turb_kw = per_turbine_power_kw(ws_arr, density)
            farm_mw = np.clip(per_turb_kw * active / 1000.0, 0.0, CAPACITY_MW)
            df[f"{label}_farm_mw"] = farm_mw
            df[f"{label}_per_turb_kw"] = per_turb_kw
            df[f"{label}_ratio"] = farm_mw / CAPACITY_MW

    # Average across wind speed sources (consensus theoretical).
    if all(c in df.columns for c in ("ds_80m_farm_mw", "ds_120m_farm_mw")):
        cols = ["ds_80m_farm_mw", "ds_120m_farm_mw"]
        if "ds_era5_100m_farm_mw" in df.columns:
            cols.append("ds_era5_100m_farm_mw")
        df["ds_consensus_farm_mw"] = df[cols].mean(axis=1)
        df["ds_consensus_ratio"] = df["ds_consensus_farm_mw"] / CAPACITY_MW

    return df
