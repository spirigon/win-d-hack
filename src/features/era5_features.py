"""Advanced ERA5-derived features.

Building on the ERA5 breakthrough, add more physics-informed features
computed from ERA5 reanalysis:
- ERA5-based v_eff, REWS proxy, power density
- ERA5-based air density using ERA5 temperature and pressure
- ERA5 wind shear (10m vs 100m)
- ERA5 gust turbulence intensity
- NWP-vs-ERA5 residuals per variable (bias signatures)
- ERA5 power curve features (sector-specific)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.features.physics import (
    IsotonicPowerCurve,
    SectorIsotonicPowerCurve,
    compute_air_density,
    compute_v_eff,
    compute_wpd,
    fit_sector_isotonic,
)


def add_era5_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced ERA5-derived features to a dataframe with base ERA5 merged."""
    df = df.copy()

    # ERA5 air density (using ERA5 pressure + temperature).
    df["era5_air_density"] = compute_air_density(df["era5_pressure_msl"], df["era5_temperature_2m"])
    df["era5_density_ratio"] = df["era5_air_density"] / 1.225

    # ERA5 v_eff at 100m (hub proxy, density-corrected).
    df["era5_v_eff"] = compute_v_eff(df["era5_wind_speed_100m"], df["era5_air_density"])
    df["era5_v_eff_cube"] = df["era5_v_eff"] ** 3
    df["era5_v_eff_cube_x_active"] = df["era5_v_eff_cube"] * df["active_turbines_ratio"]

    # ERA5 WPD.
    df["era5_wpd"] = compute_wpd(df["era5_wind_speed_100m"], df["era5_air_density"])
    df["era5_wpd_x_active"] = df["era5_wpd"] * df["active_turbines_ratio"]

    # ERA5 wind shear (10m -> 100m).
    df["era5_shear_10_100"] = df["era5_wind_speed_100m"] - df["era5_wind_speed_10m"]
    df["era5_shear_ratio_10_100"] = df["era5_wind_speed_100m"] / (df["era5_wind_speed_10m"] + 1e-3)

    # ERA5 gust features.
    df["era5_gust_ratio_10m"] = df["era5_wind_gusts_10m"] / (df["era5_wind_speed_10m"] + 1e-3)
    df["era5_gust_excess_10m"] = (df["era5_wind_gusts_10m"] - df["era5_wind_speed_10m"]).clip(lower=0.0)

    # NWP-vs-ERA5 residuals (bias signatures at every level).
    df["ws80_vs_era5_100"] = df["wind_speed_80m"] - df["era5_wind_speed_100m"]
    df["ws120_vs_era5_100"] = df["wind_speed_120m"] - df["era5_wind_speed_100m"]
    df["ws180_vs_era5_100"] = df["wind_speed_180m"] - df["era5_wind_speed_100m"]
    df["temp_vs_era5"] = df["temperature_80m"] - df["era5_temperature_2m"]

    # Direction agreement between NWP and ERA5.
    nwp_dir_rad = np.deg2rad(df["wind_direction_120m"] * 1000.0)
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    dir_dot = (
        np.sin(nwp_dir_rad) * np.sin(era5_dir_rad)
        + np.cos(nwp_dir_rad) * np.cos(era5_dir_rad)
    )
    df["nwp_era5_dir_agreement"] = dir_dot.clip(-1.0, 1.0)
    df["nwp_era5_dir_diff"] = np.arccos(df["nwp_era5_dir_agreement"])

    # ERA5 direction sector.
    era5_dir_deg = df["era5_wind_direction_100m"].to_numpy()
    df["era5_dir_sector_8"] = (era5_dir_deg // 45).astype(int) % 8
    df["era5_dir_sector_16"] = (era5_dir_deg // 22.5).astype(int) % 16

    # ERA5 × ERA5 direction interactions.
    df["era5_v_eff_x_sector"] = df["era5_v_eff"] * df["era5_dir_sector_8"]

    # ERA5 cloud cover as percentage (0-100) normalized.
    df["era5_cloud_cover_norm"] = df["era5_cloud_cover_low"] / 100.0

    # Precipitation agreement.
    df["precip_vs_era5_rain"] = df["rain"] - df["era5_rain"]

    return df


class ERA5SectorPowerCurve:
    """Power curve fit on ERA5 v_eff per direction sector."""

    def __init__(self, n_sectors: int = 8) -> None:
        self.n_sectors = n_sectors
        self.global_curve = IsotonicPowerCurve()
        self.sector_curves: list[IsotonicPowerCurve] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ERA5SectorPowerCurve":
        """Fit from a dataframe with era5_v_eff, era5_wind_direction_100m, TARGET_COL."""
        mask = df[TARGET_COL].notna() & (df[TARGET_COL] >= 0)
        data = df[mask]

        self.global_curve.fit(data["era5_v_eff"], data[TARGET_COL])

        sector_width = 360.0 / self.n_sectors
        dir_sector = (data["era5_wind_direction_100m"].to_numpy() // sector_width).astype(int) % self.n_sectors

        self.sector_curves = []
        for s in range(self.n_sectors):
            mask_s = dir_sector == s
            curve = IsotonicPowerCurve()
            if mask_s.sum() > 50:
                curve.fit(data["era5_v_eff"].iloc[mask_s], data[TARGET_COL].iloc[mask_s])
            else:
                curve = self.global_curve
            self.sector_curves.append(curve)

        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (sector_pred, global_pred)."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        v_eff = df["era5_v_eff"].to_numpy()
        dir_deg = df["era5_wind_direction_100m"].to_numpy()
        sector_width = 360.0 / self.n_sectors
        dir_sector = (dir_deg // sector_width).astype(int) % self.n_sectors

        sector_pred = np.zeros_like(v_eff, dtype=float)
        for s in range(self.n_sectors):
            mask = dir_sector == s
            if mask.any():
                sector_pred[mask] = self.sector_curves[s].predict(v_eff[mask])
        sector_pred = np.clip(sector_pred, 0.0, CAPACITY_MW)
        global_pred = self.global_curve.predict(v_eff)
        return sector_pred, global_pred


def add_era5_power_curve_features(df: pd.DataFrame, era5_pc: ERA5SectorPowerCurve) -> pd.DataFrame:
    df = df.copy()
    sector_pred, global_pred = era5_pc.predict(df)
    df["era5_p_curve_sector"] = sector_pred
    df["era5_p_curve_global"] = global_pred
    df["era5_p_curve_x_active"] = sector_pred * df["active_turbines_ratio"]
    df["era5_p_curve_ratio"] = sector_pred / CAPACITY_MW
    df["era5_p_curve_sector_minus_global"] = sector_pred - global_pred
    # Cross-agreement: do NWP and ERA5 power curves agree?
    if "p_curve_sector" in df.columns:
        df["nwp_era5_pc_diff"] = df["p_curve_sector"] - df["era5_p_curve_sector"]
    return df
