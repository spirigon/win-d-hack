"""ERA5 v2 + Open-Meteo multi-NWP ensemble loaders.

Provides two independent column groups that can be merged onto the main
training/validation frame in addition to (not in place of) the existing
`era5_reanalysis.parquet` features:

  - **ERA5v2 columns**: pressure-level winds (850 / 925 hPa), boundary-layer
    height, CAPE, low-level jet flag, soil temperature & moisture, surface
    radiation/heat fluxes, total column water vapour. All NWP-independent
    (analysis fields), full Q1 2026 coverage in the parquet.

  - **Open-Meteo NWP ensemble columns**: forecast retrospective from four
    NWPs (ECMWF IFS, GFS, ICON, OM "best_match"). Provided separately so
    the ablation can switch them on/off independently.

The functions here purposefully return only timestamp + ``era5v2_*`` /
``om_ens_*`` columns. They never touch existing column names.

Usage from a training script:

    from src.features.era5_v2 import merge_era5_v2, merge_om_ensemble
    df = merge_era5_v2(df)            # adds era5v2_* columns
    df = merge_om_ensemble(df)        # adds om_ens_* columns (optional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import TIMESTAMP_COL

_ROOT = Path(__file__).resolve().parents[2]
ERA5_V2_PATH = _ROOT / "data" / "interim" / "era5_features_v2.parquet"
OM_V2_PATH = _ROOT / "data" / "interim" / "openmeteo_features_v2.parquet"

# ERA5 v2 columns that are full-coverage in Q1 2026 and physically distinct
# from anything currently in the pipeline. Renamed with `era5v2_` prefix to
# avoid colliding with the existing `era5_*` columns produced by `_merge_era5`
# in train_v29_tuned.
_ERA5V2_KEEP: tuple[str, ...] = (
    # boundary-layer / mixing
    "era5_blh",
    "era5_cbh",
    # pressure-level winds (analysis, not forecast)
    "era5_u850", "era5_v850",
    "era5_u925", "era5_v925",
    "era5_wind_speed_850hpa",
    "era5_wind_speed_925hpa",
    "era5_llj_strength",
    "era5_is_llj",
    # convection / moisture
    "era5_cape",
    "era5_tcwv",
    "era5_hcc",
    "era5_mcc",
    # surface fluxes
    "era5_sshf",
    "era5_slhf",
    "era5_cp",
    "era5_lsp",
    # land surface
    "era5_stl1", "era5_stl4",
    "era5_swvl1", "era5_swvl2",
    "era5_snowc",
    "era5_sde",
    "era5_skt",
    # derived / pre-computed
    "era5_dewpoint_depression",
    "era5_skt_minus_t2m",
    "era5_stl1_minus_t2m",
    "era5_air_density",
    "era5_alpha_10_100",
    "era5_wind_speed_84m",
    "era5_wind_speed_120m",
    "era5_wind_eq_100m",
    "era5_has_snow",
    "era5_is_daytime",
    # land-surface 10m wind (different grid / source from regular u10/v10)
    "era5_u10_land", "era5_v10_land",
)

# Open-Meteo NWP per-source columns: 4 sources × {wind_speed_100m,
# wind_speed_10m, wind_dir_100m, gusts_10m}. Forecast retrospectives — the
# ensemble across them is a separate experiment from ERA5v2.
_OM_WIND_100M_SOURCES: tuple[str, ...] = (
    "om_wind_speed_100m_ecmwf_ifs025",
    "om_wind_speed_100m_gfs_global",
    "om_wind_speed_100m_icon_global",
    "om_wind_speed_100m_best_match",
)
_OM_WIND_10M_SOURCES: tuple[str, ...] = (
    "om_wind_speed_10m_ecmwf_ifs025",
    "om_wind_speed_10m_gfs_global",
    "om_wind_speed_10m_icon_global",
    "om_wind_speed_10m_best_match",
)
_OM_GUST_SOURCES: tuple[str, ...] = (
    "om_wind_gusts_10m_ecmwf_ifs025",
    "om_wind_gusts_10m_gfs_global",
    "om_wind_gusts_10m_icon_global",
    "om_wind_gusts_10m_best_match",
)
_OM_DIR_100M_SOURCES: tuple[str, ...] = (
    "om_wind_direction_100m_ecmwf_ifs025",
    "om_wind_direction_100m_gfs_global",
    "om_wind_direction_100m_icon_global",
    "om_wind_direction_100m_best_match",
)


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — copy the v2 drop into data/interim/")
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"{path.name} must have a 'ts' column.")
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def merge_era5_v2(
    df: pd.DataFrame,
    *,
    parquet_path: Path | str = ERA5_V2_PATH,
    timestamp_col: str = TIMESTAMP_COL,
    fillna: bool = True,
) -> pd.DataFrame:
    """Left-join ERA5 v2 columns onto ``df`` by timestamp.

    All added columns are prefixed ``era5v2_`` so they never overwrite the
    existing ``era5_*`` group merged by ``train_v29_tuned._merge_era5``.

    With ``fillna=True`` (default) any remaining NaN after the join is set
    to 0.0, mirroring the behaviour of the existing ``_merge_era5`` helper
    so LightGBM doesn't see fold-dependent NaN patterns.
    """
    era5 = _load_parquet(Path(parquet_path))

    keep = ["ts"] + [c for c in _ERA5V2_KEEP if c in era5.columns]
    era5 = era5[keep].copy()

    # Rename to era5v2_ prefix to avoid colliding with existing era5_*
    rename = {c: c.replace("era5_", "era5v2_") for c in era5.columns if c.startswith("era5_")}
    era5 = era5.rename(columns=rename)

    df = df.merge(era5, left_on=timestamp_col, right_on="ts", how="left")
    if "ts" in df.columns and "ts" != timestamp_col:
        df = df.drop(columns=["ts"])

    if fillna:
        v2_cols = [c for c in df.columns if c.startswith("era5v2_")]
        df[v2_cols] = df[v2_cols].fillna(0.0)
    return df


def merge_om_ensemble(
    df: pd.DataFrame,
    *,
    parquet_path: Path | str = OM_V2_PATH,
    timestamp_col: str = TIMESTAMP_COL,
    fillna: bool = True,
) -> pd.DataFrame:
    """Left-join Open-Meteo NWP ensemble features onto ``df``.

    Adds the following ``om_ens_*`` columns (computed here, not taken from
    the parquet, so we control exactly what enters the model):

      - ``om_ens_ws100_mean``   row-wise mean of the 4 NWP 100 m wind speeds
      - ``om_ens_ws100_median`` row-wise median
      - ``om_ens_ws100_std``    row-wise std (uncertainty proxy)
      - ``om_ens_ws100_range``  max - min across the 4 sources
      - ``om_ens_ws10_mean``    same for 10 m
      - ``om_ens_gust_mean``    same for gusts
      - ``om_ens_ws100_84m``    ensemble mean extrapolated 100 m → 84 m hub
                                 (using ISA neutral α = 0.14)
      - ``om_ens_dir_sin/cos``  circular-mean direction → sin/cos
      - ``om_ens_nwp_diff``     hackathon NWP wind_speed_120m − ensemble mean
                                 (only computed if ``wind_speed_120m`` is in df)

    Per-source columns are NOT carried through. The point is to give the
    model the ensemble summary, not 16 redundant columns.
    """
    om = _load_parquet(Path(parquet_path))

    cols_100 = ["ts"] + [c for c in _OM_WIND_100M_SOURCES if c in om.columns]
    cols_10  = [c for c in _OM_WIND_10M_SOURCES  if c in om.columns]
    cols_gst = [c for c in _OM_GUST_SOURCES      if c in om.columns]
    cols_dir = [c for c in _OM_DIR_100M_SOURCES  if c in om.columns]

    # Build the ensemble summary frame indexed by ts.
    summary = pd.DataFrame({"ts": om["ts"]})

    if len(cols_100) >= 3:  # at least 2 sources after dropping the ts entry
        ws100 = om[cols_100[1:]].astype(float)
        summary["om_ens_ws100_mean"]   = ws100.mean(axis=1, skipna=True)
        summary["om_ens_ws100_median"] = ws100.median(axis=1, skipna=True)
        summary["om_ens_ws100_std"]    = ws100.std(axis=1, skipna=True)
        summary["om_ens_ws100_range"]  = ws100.max(axis=1, skipna=True) - ws100.min(axis=1, skipna=True)
        # Extrapolate to 84 m using ISA neutral α = 0.14
        summary["om_ens_ws100_84m"]    = summary["om_ens_ws100_mean"] * (84.0 / 100.0) ** 0.14

    if len(cols_10) >= 2:
        ws10 = om[cols_10].astype(float)
        summary["om_ens_ws10_mean"] = ws10.mean(axis=1, skipna=True)

    if len(cols_gst) >= 2:
        gst = om[cols_gst].astype(float)
        summary["om_ens_gust_mean"] = gst.mean(axis=1, skipna=True)

    if len(cols_dir) >= 2:
        dirs = om[cols_dir].astype(float).values  # degrees
        rad = np.deg2rad(dirs)
        sin_mean = np.nanmean(np.sin(rad), axis=1)
        cos_mean = np.nanmean(np.cos(rad), axis=1)
        summary["om_ens_dir_sin"] = sin_mean
        summary["om_ens_dir_cos"] = cos_mean

    df = df.merge(summary, left_on=timestamp_col, right_on="ts", how="left")
    if "ts" in df.columns and "ts" != timestamp_col:
        df = df.drop(columns=["ts"])

    # Bias signal: hackathon NWP minus OM ensemble (computed AFTER merge so
    # we have wind_speed_120m available).
    if "wind_speed_120m" in df.columns and "om_ens_ws100_mean" in df.columns:
        # Project hackathon 120m down to 100m for fair comparison: v100 = v120 * (100/120)^α
        # Use a constant ISA α; this is just a diagnostic signal, not a physics claim.
        df["om_ens_nwp_diff"] = df["wind_speed_120m"] * (100.0 / 120.0) ** 0.14 - df["om_ens_ws100_mean"]

    if fillna:
        ens_cols = [c for c in df.columns if c.startswith("om_ens_")]
        df[ens_cols] = df[ens_cols].fillna(0.0)
    return df


def era5v2_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``era5v2_*`` columns currently present in ``df``."""
    return [c for c in df.columns if c.startswith("era5v2_")]


def om_ensemble_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``om_ens_*`` columns currently present in ``df``."""
    return [c for c in df.columns if c.startswith("om_ens_")]
