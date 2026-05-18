"""Extra physics-informed features worth testing.

Based on ideas that are low-risk and not redundant with existing features:
- Additional wind shear pairs (10-120, 10-180, 80-180)
- Wind vector components u, v at hub height
- Wind regime categorical
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extra shear pairs (cross-height atmospheric structure).
    df["ws_shear_10_120"] = df["wind_speed_120m"] - df["wind_speed_10m"]
    df["ws_shear_10_180"] = df["wind_speed_180m"] - df["wind_speed_10m"]
    df["ws_shear_80_180"] = df["wind_speed_180m"] - df["wind_speed_80m"]

    # Wind vector components at 120m (physical momentum).
    dir_rad_120 = np.deg2rad(df["wind_direction_120m"] * 1000.0)
    df["ws120_u"] = df["wind_speed_120m"] * np.cos(dir_rad_120)  # west-east
    df["ws120_v"] = df["wind_speed_120m"] * np.sin(dir_rad_120)  # south-north

    # Same at 80m.
    dir_rad_80 = np.deg2rad(df["wind_direction_80m"] * 1000.0)
    df["ws80_u"] = df["wind_speed_80m"] * np.cos(dir_rad_80)
    df["ws80_v"] = df["wind_speed_80m"] * np.sin(dir_rad_80)

    # Wind regime (soft, continuous — better than hard categorical).
    ws_120 = df["wind_speed_120m"]
    df["regime_calm"] = np.clip((4.0 - ws_120) / 2.0, 0, 1)  # 1 when ws<2, 0 when ws>4
    df["regime_high"] = np.clip((ws_120 - 10.0) / 4.0, 0, 1)  # 1 when ws>14, 0 when ws<10
    df["regime_normal"] = 1.0 - df["regime_calm"] - df["regime_high"]

    # ERA5 wind vector (if ERA5 is merged).
    if "era5_wind_speed_100m" in df.columns:
        era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
        df["era5_ws100_u"] = df["era5_wind_speed_100m"] * np.cos(era5_dir_rad)
        df["era5_ws100_v"] = df["era5_wind_speed_100m"] * np.sin(era5_dir_rad)

    # Bulk Richardson number (80–120 m layer) — atmospheric stability proxy.
    # Positive Ri = stable (nocturnal jet regime), Negative = unstable (convective).
    # Combines temperature gradient with wind shear in a single dimensionless number;
    # the model already sees shear and temperature separately but not their ratio.
    if "temperature_80m" in df.columns and "temperature_120m" in df.columns:
        dz = 40.0  # 120m - 80m
        T_mean = (df["temperature_80m"] + df["temperature_120m"]) / 2.0 + 273.15
        dT = df["temperature_120m"] - df["temperature_80m"]  # K
        dU = df["wind_speed_120m"] - df["wind_speed_80m"]    # m/s
        ri_raw = (9.81 / T_mean) * (dT / dz) / ((dU / dz) ** 2 + 1e-6)
        df["ri_bulk_80_120"] = np.clip(ri_raw, -10.0, 10.0)
        # Stable flag: Ri > 0.25 is conventionally the critical value
        df["is_stable_bl"] = (ri_raw > 0.0).astype(np.float32)

    return df
