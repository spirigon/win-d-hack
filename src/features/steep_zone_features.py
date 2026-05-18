"""Steep-zone (7-10 m/s) targeted features.

The 7-10 m/s regime has r=0.68 (worst correlation of any wind speed bin, 24.89% nMAE).
Power is proportional to v^3 here so 1 m/s NWP error causes ~20-40% power error.

Features added:
  - Jensen correction  : E[P(v)] > P(E[v]) in cubic regime; correction = 3*(sigma/mu)^2
  - Wind speed CV      : ens3_spread / ens3_mean (relative NWP uncertainty)
  - Cubic energy spread: |gfs_ws100m_cube - ecmwf_ws100m_cube|  (energy-domain disagreement)
  - GFS dir instability: rolling circular variance at 12h and 24h using GFS direction
  - GFS-ECMWF dir diff : angle between GFS and ECMWF wind directions
  - Steep zone interax : steep_flag * {spread, dir_stability, wind_cv, cube_spread}
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_STEEP_LO = 7.0
_STEEP_HI = 10.0


def _steep_flag(df: pd.DataFrame) -> pd.Series:
    ws = df["wind_speed_120m"].astype(float)
    return ((ws >= _STEEP_LO) & (ws <= _STEEP_HI)).astype(np.float32)


def add_jensen_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Jensen's inequality correction term for the cubic power curve.

    In the cubic regime: E[P(v)] ≈ P(mu) * (1 + 3*(sigma/mu)^2)
    The correction factor 3*(sigma/mu)^2 > 0 means expected power
    is higher than power at the mean wind speed when there is spread.
    """
    if "ens3_ws100_std" not in df.columns or "wind_speed_120m" not in df.columns:
        return df
    df = df.copy()
    spread = df["ens3_ws100_std"].astype(float)
    ws = df["wind_speed_120m"].astype(float).clip(lower=0.5)
    df["jensen_correction"] = (3.0 * (spread / ws) ** 2).astype(np.float32)

    steep = _steep_flag(df)
    df["steep_zone_flag"] = steep
    df["steep_x_jensen"] = (steep * df["jensen_correction"]).astype(np.float32)
    return df


def add_wind_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Wind speed coefficient of variation: relative NWP ensemble spread."""
    if "ens3_ws100_std" not in df.columns or "ens3_ws100_mean" not in df.columns:
        return df
    df = df.copy()
    cv = (df["ens3_ws100_std"] / df["ens3_ws100_mean"].clip(lower=0.1)).astype(np.float32)
    df["wind_cv"] = cv
    steep = _steep_flag(df)
    df["steep_x_wind_cv"] = (steep * cv).astype(np.float32)
    return df


def add_cubic_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Energy-domain (v^3) spread between GFS and ECMWF 100m wind speeds."""
    if "gfs_ws100m_cube" not in df.columns or "ecmwf_ws100m_cube" not in df.columns:
        return df
    df = df.copy()
    cube_diff = (df["gfs_ws100m_cube"] - df["ecmwf_ws100m_cube"]).abs().astype(np.float32)
    df["cube_gfs_ecmwf_diff"] = cube_diff
    steep = _steep_flag(df)
    df["steep_x_cube_diff"] = (steep * cube_diff).astype(np.float32)
    return df


def add_extended_dir_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Direction instability using GFS wind direction at 12h and 24h windows.

    Uses circular mean resultant length R ∈ [0,1]:
      R = 1 → perfectly aligned directions (stable)
      R = 0 → uniformly random directions (unstable)
    instability = 1 - R, so high instability = turbulent / variable wind direction.
    """
    df = df.copy()
    if "gfs_dir100m_sin" in df.columns and "gfs_dir100m_cos" in df.columns:
        sin_g = df["gfs_dir100m_sin"].astype(float)
        cos_g = df["gfs_dir100m_cos"].astype(float)
        for w in [12, 24]:
            mp = max(1, w // 2)
            sin_r = sin_g.rolling(window=w, min_periods=mp).mean()
            cos_r = cos_g.rolling(window=w, min_periods=mp).mean()
            R = np.sqrt(sin_r ** 2 + cos_r ** 2).clip(0.0, 1.0)
            df[f"gfs_dir_instability_{w}h"] = (1.0 - R).astype(np.float32)

    if (
        "gfs_dir100m_sin" in df.columns
        and "ecmwf_dir100m_sin" in df.columns
        and "gfs_dir100m_cos" in df.columns
        and "ecmwf_dir100m_cos" in df.columns
    ):
        dot = (
            df["gfs_dir100m_sin"].astype(float) * df["ecmwf_dir100m_sin"].astype(float)
            + df["gfs_dir100m_cos"].astype(float) * df["ecmwf_dir100m_cos"].astype(float)
        ).clip(-1.0, 1.0)
        df["gfs_ecmwf_dir_diff"] = np.arccos(dot).astype(np.float32)
    return df


def add_steep_zone_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Steep-zone flag × existing rolling spread and stability features."""
    if "wind_speed_120m" not in df.columns:
        return df
    df = df.copy()
    steep = _steep_flag(df)
    for col in [
        "ens3_spread_roll6h_mean",
        "ens3_spread_roll12h_mean",
        "dir_stability_6h",
        "gfs_dir_instability_12h",
        "gfs_ecmwf_dir_diff",
    ]:
        if col in df.columns:
            df[f"steep_x_{col}"] = (steep * df[col].astype(float)).astype(np.float32)
    return df


def add_all_steep_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all steep-zone targeted features. Call after add_all_advanced_features."""
    df = add_jensen_correction(df)
    df = add_wind_cv(df)
    df = add_cubic_spread(df)
    df = add_extended_dir_stability(df)
    df = add_steep_zone_interactions(df)
    return df


def steep_zone_columns(df: pd.DataFrame) -> list[str]:
    prefixes = (
        "jensen_correction", "steep_zone_flag", "steep_x_",
        "wind_cv", "cube_gfs_ecmwf_diff", "gfs_dir_instability_",
        "gfs_ecmwf_dir_diff",
    )
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
