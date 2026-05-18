"""Fetch MERRA-2 meteorological data from the NASA POWER API.

Source: https://power.larc.nasa.gov/api/temporal/hourly/point
Grid:   0.5° × 0.625° (~50 km), MERRA-2 reanalysis
Coords: 46.851508°N, 38.708097°E (refined farm location)

NASA POWER provides a different reanalysis product from ERA5 (MERRA-2 vs
ECMWF IFS), so the signal should provide genuine ensemble diversity.

The API hourly key format is "YYYYMMDDHHmm" (12 chars), e.g. "202201010000".

Parameters fetched (all hourly, UTC):
    WS10M     — wind speed at 10 m (m/s)
    WS50M     — wind speed at 50 m (m/s)
    WD10M     — wind direction at 10 m (°)
    WD50M     — wind direction at 50 m (°)
    T2M       — air temperature at 2 m (°C)
    PS        — surface pressure (kPa)
    SLP       — sea-level pressure (hPa)
    RH2M      — relative humidity at 2 m (%)
    PRECTOTCORR — precipitation (mm/hour)
    CLOUD_AMT — cloud amount (%)
    ALLSKY_SFC_SW_DWN — downwelling shortwave radiation (W/m²)
    Z0M       — aerodynamic roughness length (m)  ← unique vs ERA5
    DISPH     — zero-plane displacement height (m) ← unique vs ERA5

Usage:
    python scripts/fetch_nasa_power.py                         # fetch all years
    python scripts/fetch_nasa_power.py --start 2024-01-01     # specific range
    python scripts/fetch_nasa_power.py --from-json-dir data/external/nasa_raw/
                                                               # parse saved JSONs

Output:
    data/external/nasa_merra2.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LAT = 46.851508
LON = 38.708097
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

PARAMETERS = ",".join([
    "WS10M", "WS50M", "WD10M", "WD50M",
    "T2M", "PS", "SLP", "RH2M", "PRECTOTCORR",
    "CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "Z0M", "DISPH",
])

OUTPUT_PATH = _ROOT / "data" / "external" / "nasa_merra2.parquet"


def _parse_power_json(data: dict) -> pd.DataFrame:
    """Parse a NASA POWER JSON response into a tidy DataFrame.

    POWER hourly key format: 'YYYYMMDDHHmm' (12 chars, minutes always 00).
    E.g. '202201010000' = 2022-01-01 00:00 UTC.
    """
    if "properties" not in data:
        raise RuntimeError(f"Unexpected response: {list(data.keys())}")
    hourly = data["properties"]["parameter"]
    # Build dataframe from the parameter dicts.
    df = pd.DataFrame(hourly)
    # Parse index.
    idx = df.index.astype(str)
    sample = idx[0] if len(idx) else "202201010000"
    n = len(sample)
    try:
        if n == 12:
            # "YYYYMMDDHHmm"
            ts = pd.to_datetime(idx, format="%Y%m%d%H%M", errors="coerce")
        elif n == 10:
            # "YYYYMMDDHH" — no minutes
            ts = pd.to_datetime(idx, format="%Y%m%d%H", errors="coerce")
        elif n == 9:
            # "YYYYDDDHH" — day-of-year
            year_s = idx.str[:4]
            doy_s  = idx.str[4:7]
            hour_s = idx.str[7:9]
            base   = pd.to_datetime(year_s + doy_s, format="%Y%j", errors="coerce")
            ts     = base + pd.to_timedelta(hour_s.astype(int, errors="ignore"), unit="h")
        else:
            ts = pd.to_datetime(idx, infer_datetime_format=True, errors="coerce")
    except Exception:
        ts = pd.to_datetime(idx, errors="coerce")

    df.index = ts
    df.index.name = "time"
    df = df.reset_index()
    # Replace -999 fill values with NaN.
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col] > -998.0, np.nan)
    return df


def _fetch_year(year: int) -> pd.DataFrame:
    params = {
        "parameters":    PARAMETERS,
        "community":     "RE",
        "longitude":     LON,
        "latitude":      LAT,
        "start":         f"{year}0101",
        "end":           f"{year}1231",
        "format":        "JSON",
        "time-standard": "UTC",
    }
    print(f"  Fetching {year}...")
    t0 = time.time()
    try:
        r = requests.get(BASE_URL, params=params, timeout=180)
        r.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Cannot reach {BASE_URL}. Run from a terminal with internet access.\n"
            f"Error: {e}"
        ) from e
    df = _parse_power_json(r.json())
    print(f"    {len(df)} rows in {time.time() - t0:.1f}s  "
          f"ts_range: {df['time'].min()} → {df['time'].max()}  "
          f"ts_nulls: {df['time'].isna().sum()}")
    return df


def parse_from_json_dir(json_dir: Path) -> pd.DataFrame:
    """Parse JSON files already saved from the POWER API."""
    import json
    frames = []
    for f in sorted(json_dir.glob("*.json")):
        print(f"  Parsing {f.name}...")
        with open(f) as fh:
            data = json.load(fh)
        df = _parse_power_json(data)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")
    return pd.concat(frames, ignore_index=True)


def fetch_all(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        df = _fetch_year(year)
        frames.append(df)
        time.sleep(1.5)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("time").reset_index(drop=True)
    all_df = all_df.drop_duplicates(subset=["time"])
    rename = {c: f"nasa_{c.lower()}" for c in all_df.columns if c != "time"}
    return all_df.rename(columns=rename)


def compare_to_era5(nasa: pd.DataFrame) -> None:
    era5_path = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
    if not era5_path.exists():
        return
    era5 = pd.read_parquet(era5_path)
    era5["time"] = pd.to_datetime(era5["time"])
    merged = pd.merge(
        nasa[["time", "nasa_ws10m", "nasa_ws50m"]].rename(columns={"time": "time"}),
        era5[["time", "wind_speed_10m", "wind_speed_100m"]],
        on="time", how="inner",
    )
    print(f"\nNASA MERRA-2 vs ERA5 diff ({len(merged)} matched hours):")
    d10 = merged["nasa_ws10m"] - merged["wind_speed_10m"]
    print(f"  WS10M vs ERA5_10m:  mean={d10.mean():+.4f}  std={d10.std():.4f}")
    d50 = merged["nasa_ws50m"] - merged["wind_speed_100m"]
    print(f"  WS50M vs ERA5_100m: mean={d50.mean():+.4f}  std={d50.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end",   default="2026-03-31")
    ap.add_argument("--from-json-dir", type=Path, default=None,
                    help="Parse pre-saved JSON files instead of fetching")
    args = ap.parse_args()

    if args.from_json_dir:
        print(f"Parsing JSON files from {args.from_json_dir}...")
        df = parse_from_json_dir(args.from_json_dir)
        rename = {c: f"nasa_{c.lower()}" for c in df.columns if c != "time"}
        df = df.rename(columns=rename)
    else:
        start_year = pd.Timestamp(args.start).year
        end_year   = pd.Timestamp(args.end).year
        print(f"Fetching NASA POWER MERRA-2 at ({LAT}°N, {LON}°E), "
              f"years {start_year}–{end_year}")
        df = fetch_all(start_year, end_year)

    print(f"\nResult: {df.shape[0]} rows × {df.shape[1]} cols")
    valid_ts = df["time"].dropna()
    if len(valid_ts):
        print(f"  time range: {valid_ts.min()} → {valid_ts.max()}")
    ws10_col = next((c for c in df.columns if "ws10" in c), None)
    if ws10_col:
        s = df[ws10_col].dropna()
        print(f"  {ws10_col}: mean={s.mean():.2f}  std={s.std():.2f}  "
              f"nulls={df[ws10_col].isna().sum()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1e6:.2f} MB)")
    compare_to_era5(df)


if __name__ == "__main__":
    main()
