"""GEM Global and ICON-Global inter-model disagreement features.

Key insight: ERA5 and ECMWF IFS in this dataset are byte-identical — our
"4-model consensus" has only 3 distinct NWP sources.  GEM (Canadian MC) and
ICON-Global provide genuine independent forecasts at 10m height (~78-79%
temporal coverage), covering Nov 2022 onwards.

These models share different systematic biases from ERA5/GFS/ICON-EU, so their
disagreement with ERA5 is a valid uncertainty / regime signal — even though we
cannot obtain their 100m wind directly.

Features added (~12):
  gem_ws10m                : GEM near-surface wind (direct model output)
  icong_ws10m              : ICON-Global near-surface wind
  gem_ws10_vs_era5         : surface wind bias GEM − ERA5  (uncertainty signal)
  icong_ws10_vs_era5       : surface wind bias ICON-G − ERA5
  gem_ws100_extrap         : GEM wind extrapolated to 100m via site-mean alpha
  icong_ws100_extrap       : ICON-G wind extrapolated to 100m
  gem_ws100_extrap_cube    : GEM^3 — power-proportional from GEM perspective
  gem_vs_era5_100m_bias    : bias of extrapolated GEM vs ERA5 at 100m
  icong_vs_era5_100m_bias  : bias of extrapolated ICON-G vs ERA5 at 100m
  surface_model_spread_10m : std(ERA5, GFS, GEM, ICON-G) ws10m — uncertainty
  surface_model_spread_100m: std of extrapolated 100m winds across models
  gem_gust_vs_era5         : GEM gust bias vs ERA5 gust (TI uncertainty signal)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
GEM_PATH   = _ROOT / "data" / "external" / "gem_global.parquet"
ICONG_PATH = _ROOT / "data" / "external" / "icon_global.parquet"

# Site-specific mean Hellmann shear exponent: ln(ws100/ws10) / ln(100/10)
# Computed from ERA5: mean=0.200 across full dataset
SITE_ALPHA_MEAN = 0.200
LOG_RATIO_10_100 = np.log(100.0 / 10.0)   # ≈ 2.303


def _extrapolate_10m_to_100m(ws10: np.ndarray, alpha: float = SITE_ALPHA_MEAN) -> np.ndarray:
    """Power-law extrapolation of 10m wind to 100m.  ws100 = ws10 × (100/10)^alpha."""
    return ws10 * (100.0 / 10.0) ** alpha


def merge_gem_icon(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    gem_path: Path = GEM_PATH,
    icong_path: Path = ICONG_PATH,
) -> pd.DataFrame:
    """Load GEM + ICON-Global and left-join onto df by timestamp."""
    if not gem_path.exists() or not icong_path.exists():
        return df

    gem   = pd.read_parquet(gem_path)[["time", "wind_speed_10m", "wind_gusts_10m",
                                        "pressure_msl"]].rename(columns={"time": timestamp_col})
    icong = pd.read_parquet(icong_path)[["time", "wind_speed_10m"]].rename(columns={"time": timestamp_col})

    gem.columns   = [timestamp_col] + [f"gem_{c}"   for c in gem.columns[1:]]
    icong.columns = [timestamp_col] + [f"icong_{c}" for c in icong.columns[1:]]

    df = df.merge(gem,   on=timestamp_col, how="left")
    df = df.merge(icong, on=timestamp_col, how="left")
    return df


def add_gem_icon_features(
    df: pd.DataFrame,
    timestamp_col: str = "METEOFORECASTHOUR_OPENM_Datetime",
) -> pd.DataFrame:
    """Build inter-model disagreement and diversity features from GEM + ICON-G."""
    df = df.copy()

    # If already present (re-run scenario), skip merge
    if "gem_wind_speed_10m" not in df.columns:
        df = merge_gem_icon(df, timestamp_col=timestamp_col)

    # Check that at least one source merged successfully
    has_gem   = "gem_wind_speed_10m"   in df.columns
    has_icong = "icong_wind_speed_10m" in df.columns
    has_era5  = "wind_speed_10m"       in df.columns
    has_gfs   = "gfs_ws100m"           in df.columns  # from multi_nwp

    if not (has_gem or has_icong):
        return df

    if has_gem:
        gem10 = df["gem_wind_speed_10m"].to_numpy(dtype=np.float64)
        df["gem_ws10m"] = gem10.astype(np.float32)

        gem100 = _extrapolate_10m_to_100m(gem10)
        df["gem_ws100_extrap"] = gem100.astype(np.float32)
        df["gem_ws100_extrap_cube"] = (gem100 ** 3).astype(np.float32)

        if has_era5:
            era10  = df["wind_speed_10m"].to_numpy(dtype=np.float64)
            era100 = df["wind_speed_100m"].to_numpy(dtype=np.float64) if "wind_speed_100m" in df.columns else np.full(len(df), np.nan)
            df["gem_ws10_vs_era5"]      = (gem10  - era10).astype(np.float32)
            df["gem_vs_era5_100m_bias"] = (gem100 - era100).astype(np.float32)

        if "gem_wind_gusts_10m" in df.columns:
            if has_era5 and "wind_gusts_10m" in df.columns:
                df["gem_gust_vs_era5"] = (
                    df["gem_wind_gusts_10m"] - df["wind_gusts_10m"]
                ).astype(np.float32)

    if has_icong:
        icong10 = df["icong_wind_speed_10m"].to_numpy(dtype=np.float64)
        df["icong_ws10m"] = icong10.astype(np.float32)

        icong100 = _extrapolate_10m_to_100m(icong10)
        df["icong_ws100_extrap"] = icong100.astype(np.float32)

        if has_era5:
            era10  = df["wind_speed_10m"].to_numpy(dtype=np.float64)
            era100 = df["wind_speed_100m"].to_numpy(dtype=np.float64) if "wind_speed_100m" in df.columns else np.full(len(df), np.nan)
            df["icong_ws10_vs_era5"]       = (icong10  - era10).astype(np.float32)
            df["icong_vs_era5_100m_bias"]  = (icong100 - era100).astype(np.float32)

    # Multi-model spread across 4 sources (uncertainty signal)
    ws10_sources = []
    if has_era5:
        ws10_sources.append(df["wind_speed_10m"].to_numpy(dtype=np.float64))
    if "gfs_ws10m" in df.columns:
        ws10_sources.append(df["gfs_ws10m"].to_numpy(dtype=np.float64))
    if has_gem:
        ws10_sources.append(df["gem_wind_speed_10m"].to_numpy(dtype=np.float64))
    if has_icong:
        ws10_sources.append(df["icong_wind_speed_10m"].to_numpy(dtype=np.float64))

    if len(ws10_sources) >= 2:
        stack10 = np.stack(ws10_sources, axis=1)
        df["surface_model_spread_10m"] = np.nanstd(stack10, axis=1).astype(np.float32)

    ws100_sources = []
    if "wind_speed_100m" in df.columns:
        ws100_sources.append(df["wind_speed_100m"].to_numpy(dtype=np.float64))
    if "gfs_ws100m" in df.columns:
        ws100_sources.append(df["gfs_ws100m"].to_numpy(dtype=np.float64))
    if has_gem:
        ws100_sources.append(_extrapolate_10m_to_100m(df["gem_wind_speed_10m"].to_numpy(dtype=np.float64)))
    if has_icong:
        ws100_sources.append(_extrapolate_10m_to_100m(df["icong_wind_speed_10m"].to_numpy(dtype=np.float64)))

    if len(ws100_sources) >= 2:
        stack100 = np.stack(ws100_sources, axis=1)
        df["surface_model_spread_100m"] = np.nanstd(stack100, axis=1).astype(np.float32)

    return df


def gem_icon_columns(df: pd.DataFrame) -> list[str]:
    targets = {
        "gem_ws10m", "icong_ws10m",
        "gem_ws10_vs_era5", "icong_ws10_vs_era5",
        "gem_ws100_extrap", "icong_ws100_extrap",
        "gem_ws100_extrap_cube",
        "gem_vs_era5_100m_bias", "icong_vs_era5_100m_bias",
        "surface_model_spread_10m", "surface_model_spread_100m",
        "gem_gust_vs_era5",
    }
    return [c for c in df.columns if c in targets]
