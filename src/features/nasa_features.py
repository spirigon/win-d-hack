"""NASA POWER MERRA-2 reanalysis features.

MERRA-2 is a completely independent reanalysis from ERA5 (NASA vs ECMWF),
providing genuine ensemble diversity and two unique physical signals not
available in any other source in the pipeline:

    nasa_z0m    aerodynamic roughness length (m)     ← UNIQUE
    nasa_disph  zero-plane displacement height (m)   ← UNIQUE

Coverage: 2022-01-01 – 2026-05-16, 100% match on train and valid timestamps.
Source:   https://power.larc.nasa.gov  (MERRA-2 reanalysis, 0.5°×0.625° grid)

Features added (all prefixed nasa_):
    Raw (pass-through, renamed for clarity):
        nasa_ws10m, nasa_ws50m           wind speed at 10 / 50 m
        nasa_wd10m_sin/cos               wind direction at 10 m (decomposed)
        nasa_wd50m_sin/cos               wind direction at 50 m (decomposed)
        nasa_t2m, nasa_ps, nasa_rh2m     temperature, pressure, humidity
        nasa_prectot, nasa_cloud_amt     precipitation, cloud cover
        nasa_allsky_sw, nasa_z0m, nasa_disph

    Derived:
        nasa_ws50m_cube                  dominant power term
        nasa_shear_10_50                 vertical shear 10→50 m
        nasa_roughness_x_ws50           Z0M × ws50 (surface drag × wind)
        nasa_disph_x_ws50               displacement height × wind
        nasa_ws50_vs_era5_100           MERRA-2 50m vs ERA5 100m (bias signal)
        nasa_ws10_vs_era5_10            MERRA-2 10m vs ERA5 10m (bias signal)
        nasa_t2m_vs_era5                MERRA-2 vs ERA5 2m temperature bias
        nwp_spread_nasa_era5            std(ERA5_100m, MERRA2_50m) — uncertainty
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
NASA_PATH = _ROOT / "data" / "external" / "nasa_merra2.parquet"

TIMESTAMP_COL = "METEOFORECASTHOUR_OPENM_Datetime"
ERA5_WS10_COL = "era5_wind_speed_10m"
ERA5_WS100_COL = "era5_wind_speed_100m"
ERA5_T2M_COL = "era5_temperature_2m"


def merge_nasa_merra2(
    df: pd.DataFrame,
    nasa_path: Path | None = None,
    fillna: bool = True,
) -> pd.DataFrame:
    """Left-join NASA MERRA-2 features onto df by timestamp.

    All added columns are prefixed ``nasa_`` to avoid collisions.
    With ``fillna=True`` (default), remaining NaN after the join are set to 0.0
    so LightGBM never fires a NaN-branch split on these features.
    """
    p = nasa_path or NASA_PATH
    if not p.exists():
        return df

    nasa = pd.read_parquet(p)
    nasa["time"] = pd.to_datetime(nasa["time"])

    df = df.copy()
    df = df.merge(nasa, left_on=TIMESTAMP_COL, right_on="time", how="left")
    df = df.drop(columns=["time"], errors="ignore")

    # --- direction decomposition (avoid circular-feature issues) ---
    for height, src_col in (("10m", "nasa_wd10m"), ("50m", "nasa_wd50m")):
        if src_col in df.columns:
            rad = np.deg2rad(df[src_col].astype(float))
            df[f"nasa_wd{height}_sin"] = np.sin(rad)
            df[f"nasa_wd{height}_cos"] = np.cos(rad)

    # --- wind speed derived ---
    if "nasa_ws50m" in df.columns:
        ws50 = df["nasa_ws50m"].astype(float)
        df["nasa_ws50m_cube"] = ws50 ** 3

        if "nasa_ws10m" in df.columns:
            df["nasa_shear_10_50"] = ws50 - df["nasa_ws10m"].astype(float)

        if "nasa_z0m" in df.columns:
            df["nasa_roughness_x_ws50"] = df["nasa_z0m"].astype(float) * ws50

        if "nasa_disph" in df.columns:
            df["nasa_disph_x_ws50"] = df["nasa_disph"].astype(float) * ws50

        # MERRA-2 vs ERA5 bias signals (model disagreement = corrective signal)
        if ERA5_WS100_COL in df.columns:
            df["nasa_ws50_vs_era5_100"] = ws50 - df[ERA5_WS100_COL].astype(float)
            ws_arr = df[[ERA5_WS100_COL, "nasa_ws50m"]].to_numpy(dtype=np.float32)
            df["nwp_spread_nasa_era5"] = np.nanstd(ws_arr, axis=1)

        if ERA5_WS10_COL in df.columns and "nasa_ws10m" in df.columns:
            df["nasa_ws10_vs_era5_10"] = (
                df["nasa_ws10m"].astype(float) - df[ERA5_WS10_COL].astype(float)
            )

    if ERA5_T2M_COL in df.columns and "nasa_t2m" in df.columns:
        df["nasa_t2m_vs_era5"] = df["nasa_t2m"].astype(float) - df[ERA5_T2M_COL].astype(float)

    # Rename verbose column names for brevity
    rename = {
        "nasa_prectotcorr": "nasa_prectot",
        "nasa_allsky_sfc_sw_dwn": "nasa_allsky_sw",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if fillna:
        nasa_cols = [c for c in df.columns if c.startswith("nasa_") or c == "nwp_spread_nasa_era5"]
        df[nasa_cols] = df[nasa_cols].fillna(0.0)

    return df


def nasa_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns added by merge_nasa_merra2."""
    return [
        c for c in df.columns
        if c.startswith("nasa_") or c == "nwp_spread_nasa_era5"
    ]
