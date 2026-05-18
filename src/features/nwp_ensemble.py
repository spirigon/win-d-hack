"""ICON-EU ensemble features.

ICON-EU (ECMWF's European mesoscale NWP at 7km) provides an independent
wind estimate with corr=0.91 vs ERA5 and std_diff=1.41 m/s — useful
ensemble diversity for correcting ERA5 systematic biases.

Note: Open-Meteo's 'ecmwf_ifs' model returns ERA5 data (100% identical);
only ICON-EU provides genuine model diversity.

Coverage: 2023-01-01 onward. 2022 rows are NaN — LightGBM handles these
natively via its NaN-branch splitting mechanism.

Features added:
    icon_ws100m          ICON-EU wind speed at 100m (hub height)
    icon_dir100m_sin/cos ICON-EU direction components
    icon_ws10m           ICON surface wind (for cross-model shear estimate)
    icon_delta_ws100m    ICON - ERA5 wind speed (model spread proxy)
    icon_ratio_ws100m    ICON / ERA5 wind speed ratio
    nwp_spread_ws100m    std(ERA5, ICON) wind speed — uncertainty proxy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
ICON_PATH = _ROOT / "data" / "external" / "icon_eu.parquet"

TIMESTAMP_COL = "METEOFORECASTHOUR_OPENM_Datetime"
ERA5_WS_COL   = "era5_wind_speed_100m"


def merge_nwp_ensemble(
    df: pd.DataFrame,
    icon_path: Path | None = None,
    impute_missing: bool = False,
) -> pd.DataFrame:
    """Merge ICON-EU features into df.

    ICON-EU parquet uses UTC hourly timestamps. 2022 rows have NaN because
    ICON-EU coverage starts 2023-01-01.

    Args:
        impute_missing: If True, fill 2022 NaN rows with neutral values
            (icon_ws = ERA5_ws, delta = 0, ratio = 1) so LightGBM never
            creates a NaN-branch split. If False (default), NaN rows are
            left as NaN and LGBM handles them via NaN-branch splitting.
    """
    icon_p = icon_path or ICON_PATH
    if not icon_p.exists():
        return df

    df = df.copy()
    icon = pd.read_parquet(icon_p)
    icon = icon.rename(columns={
        "wind_speed_100m":     "icon_ws100m",
        "wind_direction_100m": "icon_dir100m",
        "wind_speed_10m":      "icon_ws10m",
        "wind_gusts_10m":      "icon_gusts10m",
        "temperature_2m":      "icon_t2m",
        "pressure_msl":        "icon_msl",
    })
    icon = icon[["time"] + [c for c in icon.columns if c.startswith("icon_")]]

    df = df.merge(icon, left_on=TIMESTAMP_COL, right_on="time", how="left")
    df = df.drop(columns=["time"], errors="ignore")

    # Direction sin/cos decomposition (avoid circular feature issues).
    if "icon_dir100m" in df.columns:
        rad = np.deg2rad(df["icon_dir100m"])
        df["icon_dir100m_sin"] = np.sin(rad)
        df["icon_dir100m_cos"] = np.cos(rad)

    # Model spread features — require ERA5 wind column added by _merge_era5.
    if ERA5_WS_COL in df.columns and "icon_ws100m" in df.columns:
        if impute_missing:
            # Fill 2022 NaN rows with neutral values: ICON = ERA5 (no model disagreement).
            # Eliminates NaN-branch splitting overhead for pre-2023 data.
            era5_ws = df[ERA5_WS_COL].to_numpy()
            icon_ws_arr = df["icon_ws100m"].to_numpy()
            nan_mask = np.isnan(icon_ws_arr)
            if nan_mask.any():
                icon_ws_arr[nan_mask] = era5_ws[nan_mask]
                df["icon_ws100m"] = icon_ws_arr
                # Also fill icon_ws10m and other raw columns with ERA5 proxies.
                if "icon_ws10m" in df.columns:
                    era5_ws10 = df.get("wind_speed_10m", df[ERA5_WS_COL] * 0.75)
                    ws10 = df["icon_ws10m"].to_numpy()
                    ws10[nan_mask] = era5_ws10.to_numpy()[nan_mask]
                    df["icon_ws10m"] = ws10
                for col in ["icon_gusts10m", "icon_t2m", "icon_msl"]:
                    if col in df.columns:
                        arr = df[col].to_numpy()
                        arr[nan_mask] = np.nanmedian(arr)
                        df[col] = arr
                if "icon_dir100m_sin" in df.columns:
                    for dc in ["icon_dir100m_sin", "icon_dir100m_cos", "icon_dir100m"]:
                        if dc in df.columns:
                            arr = df[dc].to_numpy()
                            arr[nan_mask] = 0.0
                            df[dc] = arr

        df["icon_delta_ws100m"] = df["icon_ws100m"] - df[ERA5_WS_COL]
        df["icon_ratio_ws100m"] = df["icon_ws100m"] / (df[ERA5_WS_COL].clip(lower=0.1))
        ws_arr = df[[ERA5_WS_COL, "icon_ws100m"]].to_numpy(dtype=np.float32)
        df["nwp_spread_ws100m"] = np.nanstd(ws_arr, axis=1)
        if impute_missing:
            df["nwp_spread_ws100m"] = df["nwp_spread_ws100m"].fillna(0.0)

    return df


def nwp_ensemble_columns(df: pd.DataFrame) -> list[str]:
    """Return list of columns added by merge_nwp_ensemble."""
    return [c for c in df.columns if c.startswith("icon_") or c.startswith("nwp_")]


# ---------------------------------------------------------------------------
# Open-Meteo multi-NWP ensemble from era5_features_v2.parquet
# ---------------------------------------------------------------------------

_ERA5V2_PATH = Path(__file__).resolve().parents[2] / "data" / "interim" / "era5_features_v2.parquet"

# Columns to pull from era5v2 parquet (all have ~0% NaN in 2022-2026 Q1)
_OM_V2_COLS = (
    "om_wind_speed_100m_best_match",
    "om_wind_speed_100m_gfs_global",
    "om_wind_std",
    "om_wind_range",
    "om_wind_hub_mean",
)


def merge_om_v2_features(
    df: pd.DataFrame,
    parquet_path: Path | None = None,
) -> pd.DataFrame:
    """Add best_match / GFS NWP ensemble features from era5_features_v2.parquet.

    Unlike ICON-EU (which has 2022 NaN), these columns have full coverage for
    2022-2026 Q1 (NaN only in Apr-May 2026 which is beyond the test period).
    Post-merge NaN are filled with 0 so LightGBM never fires a NaN branch.

    Derived features added:
        om_bm_ws100      best_match wind at 100m
        om_gfs_ws100     GFS wind at 100m
        om_bm_delta      best_match - ERA5 (key corrective signal; corr=0.09 with CF residual)
        om_gfs_delta     GFS - ERA5
        om_bm_ratio      best_match / ERA5 ratio
        om_bm_gfs_diff   best_match - GFS (NWP model spread)
        om_wind_std_v2   multi-model wind std (uncertainty proxy)
        om_wind_range_v2 multi-model wind range
        om_wind_hub_v2   multi-model hub-height mean
    """
    p = parquet_path or _ERA5V2_PATH
    if not p.exists():
        return df

    era5v2 = pd.read_parquet(p, columns=["ts"] + list(_OM_V2_COLS))
    era5v2["ts"] = pd.to_datetime(era5v2["ts"])

    df = df.copy()
    df = df.merge(era5v2, left_on=TIMESTAMP_COL, right_on="ts", how="left")
    df = df.drop(columns=["ts"], errors="ignore")

    bm = df["om_wind_speed_100m_best_match"]
    gfs = df["om_wind_speed_100m_gfs_global"]
    era5_ws = df.get(ERA5_WS_COL)

    df["om_bm_ws100"]    = bm
    df["om_gfs_ws100"]   = gfs
    df["om_wind_std_v2"] = df["om_wind_std"]
    df["om_wind_range_v2"] = df["om_wind_range"]
    df["om_wind_hub_v2"] = df["om_wind_hub_mean"]

    if era5_ws is not None:
        df["om_bm_delta"]    = bm - era5_ws
        df["om_gfs_delta"]   = gfs - era5_ws
        df["om_bm_ratio"]    = bm / era5_ws.clip(lower=0.1)

    df["om_bm_gfs_diff"] = bm - gfs

    # Fill any residual NaN (post-test dates) with 0 so no NaN-branch overhead
    om_v2_new = om_v2_columns(df)
    for c in om_v2_new:
        if df[c].isna().any():
            df[c] = df[c].fillna(0.0)

    return df


def om_v2_columns(df: pd.DataFrame) -> list[str]:
    """Return list of columns added by merge_om_v2_features."""
    return [c for c in df.columns if c.startswith("om_bm_") or c.startswith("om_gfs_")
            or c in ("om_wind_std_v2", "om_wind_range_v2", "om_wind_hub_v2")]
