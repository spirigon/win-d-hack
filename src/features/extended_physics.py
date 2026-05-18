"""Extended physics features: boundary layer, heat fluxes, soil, humidity,
upper wind profile, convection, and roughness/terrain.

All features are float32. Source columns are checked individually before use;
groups that lack their primary source are skipped entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TIMESTAMP_COL

_HUB_M: float = 85.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circ_diff(a_deg: pd.Series, b_deg: pd.Series) -> pd.Series:
    """Circular difference a - b in [-180, 180] degrees."""
    diff = a_deg.astype(float) - b_deg.astype(float)
    return np.arctan2(
        np.sin(np.radians(diff)),
        np.cos(np.radians(diff)),
    ) * 180.0 / np.pi


# ---------------------------------------------------------------------------
# Group 1 — Boundary Layer Dynamics
# ---------------------------------------------------------------------------

def _add_blh_features(df: pd.DataFrame) -> pd.DataFrame:
    if "era5v2_blh" not in df.columns:
        return df

    blh = df["era5v2_blh"].astype(float)

    df["blh_hub_ratio"] = (blh / _HUB_M).clip(upper=50.0).astype(np.float32)
    df["blh_minus_hub"] = (blh - _HUB_M).clip(0.0, 3000.0).astype(np.float32)
    df["blh_log"] = np.log1p(blh).astype(np.float32)

    if "wind_speed_120m" in df.columns:
        ws120 = df["wind_speed_120m"].astype(float)
        df["blh_x_ws120"] = (blh * ws120).astype(np.float32)

    if "ws_shear_80_180" in df.columns:
        df["blh_x_shear"] = (blh * df["ws_shear_80_180"].astype(float)).astype(np.float32)

    if "era5v2_is_daytime" in df.columns:
        df["blh_time_of_day"] = (blh * df["era5v2_is_daytime"].astype(float)).astype(np.float32)

    if "era5v2_cbh" in df.columns:
        df["cbh_vs_blh"] = (
            (df["era5v2_cbh"].astype(float) - blh).clip(-2000.0, 2000.0)
        ).astype(np.float32)

    if "ri_bulk_80_120" in df.columns:
        df["blh_stability_coupled"] = (
            blh * df["ri_bulk_80_120"].astype(float)
        ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Group 2 — Surface Heat Fluxes
# ---------------------------------------------------------------------------

def _add_heat_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    if "era5v2_sshf" not in df.columns or "era5v2_slhf" not in df.columns:
        return df

    sshf = df["era5v2_sshf"].astype(float)
    slhf = df["era5v2_slhf"].astype(float)

    if "wind_speed_120m" in df.columns:
        denom = df["wind_speed_120m"].astype(float).abs() + 1.0
        df["sshf_norm"] = (sshf / denom).astype(np.float32)
        df["slhf_norm"] = (slhf / denom).astype(np.float32)
        df["heat_flux_x_ws120"] = (sshf * df["wind_speed_120m"].astype(float)).astype(np.float32)
    else:
        df["sshf_norm"] = (sshf / 1.0).astype(np.float32)
        df["slhf_norm"] = (slhf / 1.0).astype(np.float32)

    df["heat_flux_total"] = (sshf + slhf).astype(np.float32)
    df["bowen_ratio"] = (sshf / (slhf.abs() + 1.0)).astype(np.float32)
    df["flux_stability_flag"] = (sshf > 0.0).astype(np.float32)

    if "rews" in df.columns:
        df["sshf_x_rews"] = (sshf * df["rews"].astype(float)).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Group 3 — Soil Temperature Features
# ---------------------------------------------------------------------------

def _add_soil_features(df: pd.DataFrame) -> pd.DataFrame:
    if "era5v2_stl1" not in df.columns:
        return df

    stl1 = df["era5v2_stl1"].astype(float)

    if "era5v2_stl4" in df.columns:
        df["stl_deep_gradient"] = (stl1 - df["era5v2_stl4"].astype(float)).astype(np.float32)

    if "wind_speed_120m" in df.columns:
        ws120 = df["wind_speed_120m"].astype(float)
        df["stl1_x_ws120"] = (stl1 * ws120).astype(np.float32)

    if "temperature_120m" in df.columns:
        t120_k = df["temperature_120m"].astype(float) + 273.15
        df["stl1_vs_t120m"] = (stl1 - t120_k).astype(np.float32)

    if "era5v2_swvl1" in df.columns and "wind_speed_120m" in df.columns:
        df["soil_moisture_x_ws"] = (
            df["era5v2_swvl1"].astype(float) * df["wind_speed_120m"].astype(float)
        ).astype(np.float32)

    if "era5v2_snowc" in df.columns:
        df["snow_x_stl"] = (df["era5v2_snowc"].astype(float) * stl1).astype(np.float32)

    if "era5v2_swvl1" in df.columns and "era5v2_swvl2" in df.columns:
        df["soil_wetness_ratio"] = (
            df["era5v2_swvl1"].astype(float) / (df["era5v2_swvl2"].astype(float) + 0.01)
        ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Group 4 — Humidity Features
# ---------------------------------------------------------------------------

def _add_humidity_features(df: pd.DataFrame) -> pd.DataFrame:
    if "nasa_rh2m" not in df.columns:
        return df

    rh = df["nasa_rh2m"].astype(float)

    if "wind_speed_120m" in df.columns:
        df["rh_x_ws120"] = (rh * df["wind_speed_120m"].astype(float) / 100.0).astype(np.float32)

    if "air_density" in df.columns:
        df["rh_x_density"] = (rh * df["air_density"].astype(float) / 100.0).astype(np.float32)

    if "temperature_120m" in df.columns:
        df["rh_x_temp"] = (rh * df["temperature_120m"].astype(float) / 100.0).astype(np.float32)

    if "rain" in df.columns and "showers" in df.columns:
        df["rh_x_precip"] = (
            rh * (df["rain"].astype(float) + df["showers"].astype(float)) / 100.0
        ).astype(np.float32)

    if "nasa_slp" in df.columns and "nasa_ps" in df.columns:
        # nasa_slp and nasa_ps are in Pa; result in hPa
        df["slp_vs_msl"] = (
            (df["nasa_slp"].astype(float) - df["nasa_ps"].astype(float)) / 100.0
        ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Group 5 — Upper Wind Profile & 180 m Features
# ---------------------------------------------------------------------------

def _add_upper_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    if "wind_speed_180m" not in df.columns or "wind_speed_120m" not in df.columns:
        return df

    ws180 = df["wind_speed_180m"].astype(float)
    ws120 = df["wind_speed_120m"].astype(float)

    df["ws180_vs_ws120_ratio"] = (ws180 / (ws120 + 0.1)).astype(np.float32)
    df["ws180_cube"] = (ws180 ** 3).astype(np.float32)

    # Only add 120→180 shear if not already computed upstream
    if "ws_shear_120_180" not in df.columns:
        df["ws_shear_120_180"] = (
            (ws180 - ws120) / ws120.clip(lower=0.1)
        ).astype(np.float32)

    if "wind_direction_180m" in df.columns and "wind_direction_120m" in df.columns:
        df["veer_120_180"] = _circ_diff(
            df["wind_direction_180m"], df["wind_direction_120m"]
        ).astype(np.float32)

    if "wind_direction_180m" in df.columns and "wind_direction_10m" in df.columns:
        df["veer_total"] = _circ_diff(
            df["wind_direction_180m"], df["wind_direction_10m"]
        ).astype(np.float32)

    if "era5v2_wind_speed_925hpa" in df.columns:
        df["ws925_vs_ws120"] = (
            df["era5v2_wind_speed_925hpa"].astype(float) / (ws120 + 0.1)
        ).astype(np.float32)

    if "era5v2_wind_speed_84m" in df.columns:
        ws84 = df["era5v2_wind_speed_84m"].astype(float)
        df["ws84_vs_ws120"] = (ws84 / (ws120 + 0.1)).astype(np.float32)
        df["ws84_era5_cube"] = (ws84 ** 3).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Group 6 — Convective & Precipitation Features
# ---------------------------------------------------------------------------

def _add_convective_features(df: pd.DataFrame) -> pd.DataFrame:
    if "era5v2_cp" not in df.columns and "era5v2_cape" not in df.columns:
        return df

    if "era5v2_cape" in df.columns and "wind_speed_120m" in df.columns:
        cape = df["era5v2_cape"].astype(float).clip(lower=0.0)
        ws120 = df["wind_speed_120m"].astype(float)
        df["cape_x_ws_cube"] = (np.sqrt(cape) * ws120 ** 3).astype(np.float32)

    if "era5v2_cp" in df.columns and "era5v2_lsp" in df.columns:
        cp = df["era5v2_cp"].astype(float)
        lsp = df["era5v2_lsp"].astype(float)
        total = cp + lsp

        df["precip_convective_ratio"] = (cp / (total + 1e-6)).astype(np.float32)
        df["total_precip_era5"] = total.astype(np.float32)

        if "wind_speed_120m" in df.columns:
            df["precip_x_wind"] = (
                total * df["wind_speed_120m"].astype(float)
            ).astype(np.float32)

    if "era5v2_tcwv" in df.columns and "temperature_120m" in df.columns:
        df["tcwv_x_t"] = (
            df["era5v2_tcwv"].astype(float) * df["temperature_120m"].astype(float)
        ).astype(np.float32)

    if "era5v2_is_llj" in df.columns and "wind_speed_120m" in df.columns:
        df["cape_flag_x_ws"] = (
            df["era5v2_is_llj"].astype(float) * df["wind_speed_120m"].astype(float)
        ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Group 7 — Roughness & Terrain
# ---------------------------------------------------------------------------

def _add_roughness_features(df: pd.DataFrame) -> pd.DataFrame:
    if "nasa_z0m" not in df.columns:
        return df

    z0m = df["nasa_z0m"].astype(float)

    df["z0m_log"] = np.log1p(z0m).astype(np.float32)

    if "wind_speed_120m" in df.columns:
        ws120 = df["wind_speed_120m"].astype(float)
        df["z0m_x_ws120_cube"] = (z0m * ws120 ** 3).astype(np.float32)

    if "nasa_disph" in df.columns:
        disph = df["nasa_disph"].astype(float)
        df["effective_hub_height"] = (
            (_HUB_M - disph).clip(10.0, _HUB_M)
        ).astype(np.float32)

        if "ws_shear_10_80" in df.columns:
            df["disph_x_shear"] = (
                disph * df["ws_shear_10_80"].astype(float)
            ).astype(np.float32)

    if "era5v2_alpha_10_100" in df.columns:
        df["z0m_x_alpha"] = (
            z0m * df["era5v2_alpha_10_100"].astype(float)
        ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALL_FEATURE_NAMES: frozenset[str] = frozenset({
    # Group 1
    "blh_hub_ratio", "blh_minus_hub", "blh_log", "blh_x_ws120",
    "blh_x_shear", "blh_time_of_day", "cbh_vs_blh", "blh_stability_coupled",
    # Group 2
    "sshf_norm", "slhf_norm", "heat_flux_total", "heat_flux_x_ws120",
    "bowen_ratio", "flux_stability_flag", "sshf_x_rews",
    # Group 3
    "stl_deep_gradient", "stl1_x_ws120", "stl1_vs_t120m",
    "soil_moisture_x_ws", "snow_x_stl", "soil_wetness_ratio",
    # Group 4
    "rh_x_ws120", "rh_x_density", "rh_x_temp", "rh_x_precip", "slp_vs_msl",
    # Group 5
    "ws180_vs_ws120_ratio", "ws180_cube", "ws_shear_120_180",
    "veer_120_180", "veer_total", "ws925_vs_ws120", "ws84_vs_ws120", "ws84_era5_cube",
    # Group 6
    "cape_x_ws_cube", "precip_convective_ratio", "total_precip_era5",
    "precip_x_wind", "tcwv_x_t", "cape_flag_x_ws",
    # Group 7
    "z0m_log", "z0m_x_ws120_cube", "effective_hub_height",
    "disph_x_shear", "z0m_x_alpha",
})


def add_extended_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extended physics feature groups to df.

    Operates on the combined (train + validation) DataFrame after all external
    data merges (ERA5v2, NASA MERRA-2) have completed. Each group is applied
    independently; a missing source column causes that group to be skipped
    without raising an error.

    Args:
        df: Combined DataFrame with base NWP columns plus era5v2_* and nasa_*
            columns already merged.

    Returns:
        DataFrame with additional float32 feature columns appended. The input
        is never modified in place.
    """
    df = df.copy()

    df = _add_blh_features(df)
    df = _add_heat_flux_features(df)
    df = _add_soil_features(df)
    df = _add_humidity_features(df)
    df = _add_upper_wind_features(df)
    df = _add_convective_features(df)
    df = _add_roughness_features(df)

    return df


def extended_physics_columns(df: pd.DataFrame) -> list[str]:
    """Return names of extended physics columns that are present in df.

    Args:
        df: DataFrame after add_extended_physics_features() has been called.

    Returns:
        Sorted list of column names produced by this module.
    """
    return sorted(c for c in df.columns if c in _ALL_FEATURE_NAMES)
