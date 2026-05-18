"""Per-direction wake correction features.

Wake effect: downstream turbines receive lower wind due to upstream wake.
Magnitude depends on wind direction (which turbines are upstream).

We fit a lookup table of empirical correction factors:
  wake_factor(direction_sector, ws_bin) = mean(actual / datasheet)

At inference, apply the correction to the datasheet prediction.
This gives the model a "corrected datasheet" feature closer to actual farm behavior.

IMPORTANT: Fit ONLY on clean training rows (no curtailment, no outliers) to
learn pure wake physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW, TARGET_COL


@dataclass
class WakeLookup:
    """Empirical wake correction factor by (direction_sector, wind_speed_bin)."""

    n_sectors: int
    ws_bins: np.ndarray
    # Shape: (n_sectors, n_ws_bins). Multiplicative factor for datasheet.
    factors: np.ndarray
    # Fallback per sector (mean across ws bins where data was sparse).
    sector_fallback: np.ndarray

    def predict_factor(
        self, dir_deg: np.ndarray, ws: np.ndarray
    ) -> np.ndarray:
        """Return multiplicative correction factor for each row."""
        sector_width = 360.0 / self.n_sectors
        dir_sector = (dir_deg // sector_width).astype(int) % self.n_sectors

        ws_idx = np.searchsorted(self.ws_bins, ws, side="right") - 1
        ws_idx = np.clip(ws_idx, 0, len(self.ws_bins) - 2)

        result = self.factors[dir_sector, ws_idx]

        # Fallback for NaN (sparse cells).
        nan_mask = np.isnan(result)
        if nan_mask.any():
            result[nan_mask] = self.sector_fallback[dir_sector[nan_mask]]
        result = np.nan_to_num(result, nan=1.0)
        return np.clip(result, 0.3, 1.3)  # physical bounds


def fit_wake_lookup(
    df: pd.DataFrame,
    n_sectors: int = 16,
    ws_bin_edges: np.ndarray | None = None,
) -> WakeLookup:
    """Fit wake correction lookup from clean training data.

    Expects columns:
    - TARGET_COL (actual power in MW)
    - ds_80m_farm_mw or ds_120m_farm_mw (datasheet theoretical)
    - wind_direction_80m (raw, 0.001-0.360)
    - wind_speed_80m
    """
    if ws_bin_edges is None:
        ws_bin_edges = np.array([0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 25], dtype=float)

    # Use ds_80m as the theoretical reference (hub proxy).
    theo_col = "ds_80m_farm_mw" if "ds_80m_farm_mw" in df.columns else "ds_120m_farm_mw"

    mask = (
        df[TARGET_COL].notna()
        & (df[TARGET_COL] > 0.5)
        & (df[theo_col] > 1.0)  # avoid division by near-zero
    )
    data = df[mask].copy()

    # Exclude curtailment events.
    ratio = data[TARGET_COL] / data[theo_col]
    # Keep only rows where actual/theoretical is in a physical range.
    ratio_ok = (ratio > 0.3) & (ratio < 1.3)
    data = data[ratio_ok]

    ws = data["wind_speed_80m"].to_numpy()
    dir_deg = data["wind_direction_80m"].to_numpy() * 1000.0
    y = data[TARGET_COL].to_numpy()
    ds = data[theo_col].to_numpy()

    sector_width = 360.0 / n_sectors
    dir_sector = (dir_deg // sector_width).astype(int) % n_sectors

    ws_idx = np.searchsorted(ws_bin_edges, ws, side="right") - 1
    ws_idx = np.clip(ws_idx, 0, len(ws_bin_edges) - 2)

    n_ws_bins = len(ws_bin_edges) - 1
    factors = np.full((n_sectors, n_ws_bins), np.nan)

    for s in range(n_sectors):
        for b in range(n_ws_bins):
            mask_cell = (dir_sector == s) & (ws_idx == b)
            if mask_cell.sum() >= 10:  # enough data for reliable estimate
                factors[s, b] = np.median(y[mask_cell] / ds[mask_cell])

    # Sector-level fallback.
    sector_fallback = np.full(n_sectors, np.nan)
    for s in range(n_sectors):
        mask_s = dir_sector == s
        if mask_s.sum() >= 50:
            sector_fallback[s] = np.median(y[mask_s] / ds[mask_s])
    # Global fallback.
    global_fallback = np.median(y / ds)
    sector_fallback = np.where(np.isnan(sector_fallback), global_fallback, sector_fallback)

    return WakeLookup(
        n_sectors=n_sectors,
        ws_bins=ws_bin_edges,
        factors=factors,
        sector_fallback=sector_fallback,
    )


def add_wake_features(df: pd.DataFrame, wake: WakeLookup) -> pd.DataFrame:
    """Add wake correction features to a dataframe."""
    df = df.copy()

    ws = df["wind_speed_80m"].to_numpy()
    dir_deg = df["wind_direction_80m"].to_numpy() * 1000.0
    wake_factor = wake.predict_factor(dir_deg, ws)
    df["wake_factor"] = wake_factor

    # Wake-corrected datasheet prediction.
    if "ds_80m_farm_mw" in df.columns:
        df["ds_wake_corrected"] = df["ds_80m_farm_mw"] * wake_factor
        df["ds_wake_corrected_ratio"] = df["ds_wake_corrected"] / CAPACITY_MW

    if "ds_consensus_farm_mw" in df.columns:
        df["ds_consensus_wake_corrected"] = df["ds_consensus_farm_mw"] * wake_factor

    # Same for ERA5 100m if present.
    if "era5_wind_speed_100m" in df.columns and "ds_era5_100m_farm_mw" in df.columns:
        era5_dir_deg = df["era5_wind_direction_100m"].to_numpy()
        era5_ws = df["era5_wind_speed_100m"].to_numpy()
        era5_wake = wake.predict_factor(era5_dir_deg, era5_ws)
        df["era5_wake_factor"] = era5_wake
        df["ds_era5_wake_corrected"] = df["ds_era5_100m_farm_mw"] * era5_wake

    return df
