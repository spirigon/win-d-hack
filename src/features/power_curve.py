"""Empirical power curve lookup (binned mean) per direction sector.

This is a simplified KDE: for each (wind_speed_bin, direction_sector) cell,
compute the mean observed power from training data. At inference time, look up
the expected power given the weather conditions.

This gives the tree model a strong monotone baseline to split on, dramatically
reducing the search space in the 5-10 m/s transition zone where errors are
largest.

IMPORTANT: This must be fit ONLY on training data to avoid leakage. The
``fit()`` method uses only rows where the target is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW, TARGET_COL


@dataclass
class PowerCurveLookup:
    """Binned empirical power curve."""

    ws_bins: np.ndarray  # bin edges for wind speed
    n_sectors: int  # number of direction sectors
    lookup: np.ndarray  # shape (n_ws_bins, n_sectors) — mean power per cell
    global_lookup: np.ndarray  # shape (n_ws_bins,) — direction-agnostic fallback

    def predict(
        self,
        ws: np.ndarray,
        dir_sector: np.ndarray,
    ) -> np.ndarray:
        """Look up expected power for given wind speed and direction sector."""
        ws_idx = np.digitize(ws, self.ws_bins) - 1
        ws_idx = np.clip(ws_idx, 0, len(self.ws_bins) - 2)
        dir_idx = np.clip(dir_sector.astype(int), 0, self.n_sectors - 1)

        result = self.lookup[ws_idx, dir_idx]

        # Fall back to global curve where sector-specific data is sparse (NaN).
        nan_mask = np.isnan(result)
        if nan_mask.any():
            result[nan_mask] = self.global_lookup[ws_idx[nan_mask]]

        # Final fallback: 0 for any remaining NaN.
        result = np.nan_to_num(result, nan=0.0)
        return np.clip(result, 0.0, CAPACITY_MW)


def fit_power_curve(
    df: pd.DataFrame,
    ws_col: str = "wind_speed_80m",
    dir_col: str = "wind_direction_80m",
    n_sectors: int = 8,
    ws_bin_width: float = 0.5,
    ws_max: float = 25.0,
) -> PowerCurveLookup:
    """Fit the empirical power curve from training data.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with target column and weather columns.
    ws_col : str
        Wind speed column to bin.
    dir_col : str
        Wind direction column (raw, in 0..0.360 encoding).
    n_sectors : int
        Number of direction sectors (default 8 = 45° each).
    ws_bin_width : float
        Width of wind speed bins in m/s.
    ws_max : float
        Maximum wind speed to consider.
    """
    # Filter to rows with valid target (exclude maintenance/curtailment outliers).
    mask = df[TARGET_COL].notna() & (df[TARGET_COL] >= 0)
    data = df[mask].copy()

    ws = data[ws_col].to_numpy(dtype=float)
    dir_deg = data[dir_col].to_numpy(dtype=float) * 1000.0  # decode to degrees
    power = data[TARGET_COL].to_numpy(dtype=float)

    # Bin edges.
    ws_bins = np.arange(0, ws_max + ws_bin_width, ws_bin_width)
    n_ws_bins = len(ws_bins) - 1

    # Direction sector.
    sector_width = 360.0 / n_sectors
    dir_sector = (dir_deg // sector_width).astype(int) % n_sectors

    # Build lookup table.
    lookup = np.full((n_ws_bins, n_sectors), np.nan)
    global_lookup = np.full(n_ws_bins, np.nan)

    ws_idx = np.digitize(ws, ws_bins) - 1
    ws_idx = np.clip(ws_idx, 0, n_ws_bins - 1)

    for i in range(n_ws_bins):
        ws_mask = ws_idx == i
        if ws_mask.sum() > 5:
            global_lookup[i] = np.median(power[ws_mask])
        for j in range(n_sectors):
            cell_mask = ws_mask & (dir_sector == j)
            if cell_mask.sum() > 3:
                lookup[i, j] = np.median(power[cell_mask])

    return PowerCurveLookup(
        ws_bins=ws_bins,
        n_sectors=n_sectors,
        lookup=lookup,
        global_lookup=global_lookup,
    )
