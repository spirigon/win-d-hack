"""Diurnal (hour-of-day) features for wind power forecasting.

Physical motivation:
  The boundary layer has a strong diurnal cycle driven by solar heating:
  - Daytime: convective mixing → well-mixed profile, weaker low-level shear
  - Night:   stable stratification → nocturnal low-level jet → strong shear at ~100-200m
  - Coastal sites additionally show sea-breeze / land-breeze circulation
  Month-sin/cos captures the seasonal envelope; hour features capture the daily modulation
  within each season.  These interact: a summer noon is very different from a winter noon.

Expected features added (~10):
  hour_sin, hour_cos              : harmonic encoding of hour-of-day
  hour_sin_x_ws120                : hour × wind speed interaction
  hour_cos_x_ws120
  hour_sin_x_veff                 : hour × effective wind speed
  hour_cos_x_veff
  hour_x_wpd                      : hour_cos × wind power density proxy
  is_day                          : coarse day/night flag (0/1)
  daytime_ws_ratio                : ws120 × is_day  (daytime profile differs)
  night_ws_ratio                  : ws120 × (1 - is_day)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_diurnal_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    df = df.copy()

    if timestamp_col not in df.columns:
        return df

    ts   = pd.to_datetime(df[timestamp_col])
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(dtype=np.float64)

    theta = 2.0 * np.pi * hour / 24.0
    h_sin = np.sin(theta).astype(np.float32)
    h_cos = np.cos(theta).astype(np.float32)

    df["hour_sin"] = h_sin
    df["hour_cos"] = h_cos

    for col in ("wind_speed_120m", "v_eff"):
        if col in df.columns:
            v = df[col].to_numpy(dtype=np.float32)
            df[f"hour_sin_x_{col}"] = (h_sin * v).astype(np.float32)
            df[f"hour_cos_x_{col}"] = (h_cos * v).astype(np.float32)

    # Wind power density proxy (v³) × diurnal phase
    if "wind_speed_120m" in df.columns:
        wpd = df["wind_speed_120m"].to_numpy(dtype=np.float32) ** 3
        df["hour_x_wpd"] = (h_cos * wpd).astype(np.float32)

    # Rough day/night split (UTC; adjust offset if site is not UTC-aligned)
    # 06:00–20:00 UTC ≈ daytime for mid-latitudes; top competitors often use solar elevation
    is_day = ((hour >= 6) & (hour < 20)).astype(np.float32)
    df["is_day"] = is_day

    if "wind_speed_120m" in df.columns:
        ws = df["wind_speed_120m"].to_numpy(dtype=np.float32)
        df["daytime_ws"] = (ws * is_day).astype(np.float32)
        df["night_ws"]   = (ws * (1.0 - is_day)).astype(np.float32)

    return df


def diurnal_columns(df: pd.DataFrame) -> list[str]:
    targets = {
        "hour_sin", "hour_cos",
        "hour_sin_x_wind_speed_120m", "hour_cos_x_wind_speed_120m",
        "hour_sin_x_v_eff", "hour_cos_x_v_eff",
        "hour_x_wpd",
        "is_day", "daytime_ws", "night_ws",
    }
    return [c for c in df.columns if c in targets]
