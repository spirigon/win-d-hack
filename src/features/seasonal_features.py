"""Seasonal × wind-speed interaction features.

Power ∝ air density ∝ 1/T.  Winter months (Jan-Mar) produce ~5-6% more power
than summer at the same wind speed, but the model has no explicit signal for
this unless we encode month × wind interactions.

Features added:
  month_sin_x_veff     : sin(2π*month/12) × v_eff
  month_cos_x_veff     : cos(2π*month/12) × v_eff
  is_winter_x_veff     : (month ∈ {1,2,3}) × v_eff
  is_winter_x_ws120    : (month ∈ {1,2,3}) × wind_speed_120m
  month_x_ws120        : month × wind_speed_120m  (linear, for residual trends)
  season_x_wpd         : is_winter × v_eff^3      (winter wind power density)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TIMESTAMP_COL

_WINTER_MONTHS = {1, 2, 3}


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    if TIMESTAMP_COL not in df.columns:
        return df
    if "wind_speed_120m" not in df.columns:
        return df

    df = df.copy()
    month = df[TIMESTAMP_COL].dt.month.to_numpy(dtype=np.float32)
    ws120 = df["wind_speed_120m"].to_numpy(dtype=np.float32)

    # Circular month encoding — phase-preserving, captures Dec/Jan continuity
    month_rad = (2.0 * np.pi * month / 12.0).astype(np.float32)
    sin_m = np.sin(month_rad).astype(np.float32)
    cos_m = np.cos(month_rad).astype(np.float32)

    veff = df["v_eff"].to_numpy(dtype=np.float32) if "v_eff" in df.columns else ws120
    is_winter = np.isin(month.astype(int), list(_WINTER_MONTHS)).astype(np.float32)

    df["month_sin_x_veff"]  = (sin_m * veff).astype(np.float32)
    df["month_cos_x_veff"]  = (cos_m * veff).astype(np.float32)
    df["is_winter_x_veff"]  = (is_winter * veff).astype(np.float32)
    df["is_winter_x_ws120"] = (is_winter * ws120).astype(np.float32)
    df["month_x_ws120"]     = (month * ws120).astype(np.float32)
    df["season_x_wpd"]      = (is_winter * veff ** 3).astype(np.float32)

    return df


def seasonal_columns(df: pd.DataFrame) -> list[str]:
    names = (
        "month_sin_x_veff", "month_cos_x_veff",
        "is_winter_x_veff", "is_winter_x_ws120",
        "month_x_ws120", "season_x_wpd",
    )
    return [c for c in df.columns if c in names]
