"""Multi-NWP ensemble features: GFS, ECMWF IFS, and full consensus.

Two previously unused forecast models are merged here:
    GFS (NOAA Global Forecast System) — 100% clean, 2022-2026 Q1
    ECMWF IFS — 100% clean, 2022-2026 Q1

Both provide hub-height (100m) wind speed, which lets us build a 3-model
always-available consensus (ERA5 + GFS + ECMWF). When ICON-EU is present
this expands to a 4-model ensemble.

Key new signals:
    gfs_vs_era5_100     — forecast vs reanalysis disagreement (GFS)
    ecmwf_vs_era5_100   — forecast vs reanalysis disagreement (ECMWF)
    gfs_vs_ecmwf_100    — two independent forecast models disagreeing
    ens3_ws100_std      — 3-model spread = forecast uncertainty proxy
    hackathon_vs_ens3   — hackathon NWP vs consensus (biggest bias signal)
    ens3_ws100_mean     — consensus wind speed (de-biased estimate)

Rolling features on GFS and ECMWF capture temporal NWP evolution that is
independent of the ERA5-based rolling already in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
GFS_PATH    = _ROOT / "data" / "external" / "gfs_global.parquet"
ECMWF_PATH  = _ROOT / "data" / "external" / "ecmwf_ifs.parquet"

TIMESTAMP_COL  = "METEOFORECASTHOUR_OPENM_Datetime"
ERA5_WS100_COL = "era5_wind_speed_100m"
ERA5_WS10_COL  = "era5_wind_speed_10m"
ICON_WS100_COL = "icon_ws100m"


# ---------------------------------------------------------------------------
# GFS
# ---------------------------------------------------------------------------

def merge_gfs(
    df: pd.DataFrame,
    gfs_path: Path | None = None,
    fillna: bool = True,
) -> pd.DataFrame:
    """Merge GFS Global Forecast System features (100% coverage, 2022-2026 Q1)."""
    p = gfs_path or GFS_PATH
    if not p.exists():
        return df

    gfs = pd.read_parquet(p)
    gfs["time"] = pd.to_datetime(gfs["time"])
    gfs = gfs.rename(columns={
        "wind_speed_10m":     "gfs_ws10m",
        "wind_speed_100m":    "gfs_ws100m",
        "wind_direction_10m": "gfs_dir10m",
        "wind_direction_100m":"gfs_dir100m",
        "wind_gusts_10m":     "gfs_gust10m",
        "temperature_2m":     "gfs_t2m",
        "pressure_msl":       "gfs_msl",
    })
    gfs_cols = ["time"] + [c for c in gfs.columns if c.startswith("gfs_")]
    gfs = gfs[gfs_cols]

    df = df.copy()
    df = df.merge(gfs, left_on=TIMESTAMP_COL, right_on="time", how="left")
    df = df.drop(columns=["time"], errors="ignore")

    # Direction components (avoid circular input issues)
    if "gfs_dir100m" in df.columns:
        rad = np.deg2rad(df["gfs_dir100m"].astype(float))
        df["gfs_dir100m_sin"] = np.sin(rad)
        df["gfs_dir100m_cos"] = np.cos(rad)

    # Power term
    if "gfs_ws100m" in df.columns:
        ws = df["gfs_ws100m"].astype(float)
        df["gfs_ws100m_cube"] = ws ** 3

        # Bias signals vs ERA5 and vs hackathon NWP
        if ERA5_WS100_COL in df.columns:
            df["gfs_vs_era5_100"] = ws - df[ERA5_WS100_COL].astype(float)
            df["gfs_ratio_era5_100"] = ws / (df[ERA5_WS100_COL].clip(lower=0.1))

        if "wind_speed_120m" in df.columns:
            # Project hackathon 120m → 100m (ISA neutral α=0.14)
            nwp_100m_proxy = df["wind_speed_120m"].astype(float) * (100.0 / 120.0) ** 0.14
            df["gfs_vs_nwp_100"] = ws - nwp_100m_proxy

    # 6h / 12h wind tendency (requires sorted time — caller must ensure this)
    for col, lag in [("gfs_ws100m_trend_6h", 6), ("gfs_ws100m_trend_12h", 12)]:
        if "gfs_ws100m" in df.columns:
            df[col] = df["gfs_ws100m"].astype(float).diff(lag)

    if fillna:
        gfs_new = [c for c in df.columns if c.startswith("gfs_")]
        df[gfs_new] = df[gfs_new].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# ECMWF IFS
# ---------------------------------------------------------------------------

def merge_ecmwf_ifs(
    df: pd.DataFrame,
    ecmwf_path: Path | None = None,
    fillna: bool = True,
) -> pd.DataFrame:
    """Merge ECMWF IFS forecast features (100% coverage, 2022-2026 Q1)."""
    p = ecmwf_path or ECMWF_PATH
    if not p.exists():
        return df

    ecmwf = pd.read_parquet(p)
    ecmwf["time"] = pd.to_datetime(ecmwf["time"])
    ecmwf = ecmwf.rename(columns={
        "wind_speed_10m":     "ecmwf_ws10m",
        "wind_speed_100m":    "ecmwf_ws100m",
        "wind_direction_100m":"ecmwf_dir100m",
        "wind_gusts_10m":     "ecmwf_gust10m",
        "temperature_2m":     "ecmwf_t2m",
        "pressure_msl":       "ecmwf_msl",
    })
    ecmwf_cols = ["time"] + [c for c in ecmwf.columns if c.startswith("ecmwf_")]
    ecmwf = ecmwf[ecmwf_cols]

    df = df.copy()
    df = df.merge(ecmwf, left_on=TIMESTAMP_COL, right_on="time", how="left")
    df = df.drop(columns=["time"], errors="ignore")

    # Direction components
    if "ecmwf_dir100m" in df.columns:
        rad = np.deg2rad(df["ecmwf_dir100m"].astype(float))
        df["ecmwf_dir100m_sin"] = np.sin(rad)
        df["ecmwf_dir100m_cos"] = np.cos(rad)

    # Power term and bias signals
    if "ecmwf_ws100m" in df.columns:
        ws = df["ecmwf_ws100m"].astype(float)
        df["ecmwf_ws100m_cube"] = ws ** 3

        if ERA5_WS100_COL in df.columns:
            df["ecmwf_vs_era5_100"] = ws - df[ERA5_WS100_COL].astype(float)
            df["ecmwf_ratio_era5_100"] = ws / (df[ERA5_WS100_COL].clip(lower=0.1))

        if "wind_speed_120m" in df.columns:
            nwp_100m_proxy = df["wind_speed_120m"].astype(float) * (100.0 / 120.0) ** 0.14
            df["ecmwf_vs_nwp_100"] = ws - nwp_100m_proxy

        # GFS vs ECMWF (two independent forecast models)
        if "gfs_ws100m" in df.columns:
            df["gfs_vs_ecmwf_100"] = df["gfs_ws100m"].astype(float) - ws
            df["ecmwf_gfs_mean"] = (df["gfs_ws100m"].astype(float) + ws) / 2.0

    # 6h / 12h wind tendency
    for col, lag in [("ecmwf_ws100m_trend_6h", 6), ("ecmwf_ws100m_trend_12h", 12)]:
        if "ecmwf_ws100m" in df.columns:
            df[col] = df["ecmwf_ws100m"].astype(float).diff(lag)

    if fillna:
        fill_cols = [
            c for c in df.columns
            if c.startswith("ecmwf_") or c in ("gfs_vs_ecmwf_100", "ecmwf_gfs_mean")
        ]
        for c in fill_cols:
            df[c] = df[c].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Full multi-model consensus
# ---------------------------------------------------------------------------

def build_nwp_consensus(
    df: pd.DataFrame,
    fillna: bool = True,
) -> pd.DataFrame:
    """Build ensemble statistics across all available NWP sources at ~100m.

    Always-available 3-model consensus (ERA5 + GFS + ECMWF):
        ens3_ws100_mean, ens3_ws100_std, ens3_ws100_range
        ens3_ws100_min, ens3_ws100_max

    4-model when ICON-EU available (2023+):
        ens4_ws100_mean, ens4_ws100_std

    Key corrective signals:
        hackathon_vs_ens3   — hackathon NWP bias vs 3-model consensus
        ecmwf_vs_gfs_std    — agreement between the two major forecast models
        ens3_spread_flag    — high spread (>2 m/s) = uncertain forecast hour
    """
    df = df.copy()

    sources_100m: list[str] = []
    if ERA5_WS100_COL in df.columns:
        sources_100m.append(ERA5_WS100_COL)
    if "gfs_ws100m" in df.columns:
        sources_100m.append("gfs_ws100m")
    if "ecmwf_ws100m" in df.columns:
        sources_100m.append("ecmwf_ws100m")

    if len(sources_100m) >= 2:
        mat = df[sources_100m].astype(float).to_numpy()
        df["ens3_ws100_mean"]  = np.nanmean(mat, axis=1)
        df["ens3_ws100_std"]   = np.nanstd(mat, axis=1)
        df["ens3_ws100_range"] = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
        df["ens3_ws100_min"]   = np.nanmin(mat, axis=1)
        df["ens3_ws100_max"]   = np.nanmax(mat, axis=1)
        df["ens3_spread_flag"] = (df["ens3_ws100_std"] > 2.0).astype(np.int8)

        # Hackathon NWP bias vs 3-model consensus (strongest corrective signal)
        if "wind_speed_120m" in df.columns:
            nwp_at_100m = df["wind_speed_120m"].astype(float) * (100.0 / 120.0) ** 0.14
            df["hackathon_vs_ens3"] = nwp_at_100m - df["ens3_ws100_mean"]
            df["hackathon_ratio_ens3"] = nwp_at_100m / (df["ens3_ws100_mean"].clip(lower=0.1))

        # 6h rolling spread trend (is consensus converging or diverging?)
        df["ens3_spread_trend_6h"] = df["ens3_ws100_std"].diff(6)

    # 4-model: add ICON-EU when available
    sources_4m = sources_100m.copy()
    if ICON_WS100_COL in df.columns:
        sources_4m.append(ICON_WS100_COL)
    if len(sources_4m) > len(sources_100m):
        mat4 = df[sources_4m].astype(float).to_numpy()
        df["ens4_ws100_mean"] = np.nanmean(mat4, axis=1)
        df["ens4_ws100_std"]  = np.nanstd(mat4, axis=1)
        # ICON adds genuine NWP diversity: compare 4-model vs 3-model
        df["icon_shift_ens3"] = df["ens4_ws100_mean"] - df["ens3_ws100_mean"]

    if fillna:
        ens_cols = [c for c in df.columns if c.startswith("ens3_") or c.startswith("ens4_")
                    or c in ("hackathon_vs_ens3", "hackathon_ratio_ens3",
                             "ens3_spread_trend_6h", "icon_shift_ens3")]
        df[ens_cols] = df[ens_cols].fillna(0.0)

    return df


def multi_nwp_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns added by this module."""
    prefixes = ("gfs_", "ecmwf_", "ens3_", "ens4_")
    extras = {"gfs_vs_ecmwf_100", "ecmwf_gfs_mean", "hackathon_vs_ens3",
              "hackathon_ratio_ens3", "ens3_spread_trend_6h", "icon_shift_ens3"}
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in prefixes) or c in extras
    ]
