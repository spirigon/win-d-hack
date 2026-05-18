"""Fetch ERA5 reanalysis from Open-Meteo at refined farm coordinates.

The original ``data/external/era5_reanalysis.parquet`` was fetched at
(46.8268°N, 38.7179°E). The user supplied a more accurate location:

    46°51'49.7"N 38°42'19.1"E  →  46.8638°N, 38.7053°E

That's ~4 km north and ~1 km west — well below ERA5's 28 km native
resolution, but Open-Meteo interpolates between grid cells so we may get
a slightly different signal. This script fetches data at the new coords
and compares wind-speed differences vs the existing file so we know
whether retraining is worth the effort.

Usage:
    python scripts/fetch_era5_refined_coords.py
    python scripts/fetch_era5_refined_coords.py --start 2022-01-01 --end 2026-03-31
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

# Refined coordinates: 46°51'49.7"N 38°42'19.1"E  →  decimal degrees.
LAT_REFINED = 46.0 + 51.0 / 60.0 + 49.7 / 3600.0   # 46.8638
LON_REFINED = 38.0 + 42.0 / 60.0 + 19.1 / 3600.0   # 38.7053

# Original (used by data/external/era5_reanalysis.parquet).
LAT_ORIGINAL = 46.8268455973
LON_ORIGINAL = 38.7179393185

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables — matched to the existing file's schema so the new parquet is a
# drop-in replacement.
HOURLY_VARS = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
    "cloud_cover_low",
    "rain",
    "snowfall",
)

# Open-Meteo's wind-speed default unit is km/h. We want m/s to match the
# existing parquet (which already has m/s values).
WIND_SPEED_UNIT = "ms"

OUTPUT_PATH = _ROOT / "data" / "external" / "era5_reanalysis_v2_refined.parquet"


def _fetch_chunk(start: str, end: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch one date range from Open-Meteo's Historical Weather endpoint.

    Open-Meteo accepts ranges of any length for ERA5 backfill; we don't
    need to chunk by year. The response is JSON with a flat ``hourly``
    object whose values are aligned arrays.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": WIND_SPEED_UNIT,
        "models": "era5",
        "timezone": "UTC",
    }
    print(f"  GET {OPEN_METEO_URL} (lat={lat:.4f}, lon={lon:.4f}, "
          f"{start}..{end})")
    r = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js:
        raise RuntimeError(f"No 'hourly' in response: {js!r}")

    hourly = js["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "time"})
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_era5_refined(start: str, end: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch the full date range. Open-Meteo handles multi-year ranges."""
    print(f"Fetching ERA5 from Open-Meteo at ({lat:.4f}, {lon:.4f}), "
          f"{start} → {end}...")
    t0 = time.time()
    df = _fetch_chunk(start, end, lat, lon)
    elapsed = time.time() - t0
    print(f"  Got {len(df)} hourly rows in {elapsed:.1f}s")
    return df


def compare_to_existing(new_df: pd.DataFrame) -> None:
    """Compare wind speeds between the new parquet and the existing one."""
    existing_path = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
    if not existing_path.exists():
        print(f"  (no existing {existing_path.name} to compare against)")
        return
    existing = pd.read_parquet(existing_path)
    existing["time"] = pd.to_datetime(existing["time"])
    merged = pd.merge(
        new_df[["time", "wind_speed_100m", "wind_speed_10m", "pressure_msl",
                "temperature_2m", "wind_direction_100m"]],
        existing[["time", "wind_speed_100m", "wind_speed_10m", "pressure_msl",
                  "temperature_2m", "wind_direction_100m"]],
        on="time", suffixes=("_new", "_old"),
    )
    print(f"\nDiff stats vs existing file ({len(merged)} matched hours):")
    for col in ("wind_speed_100m", "wind_speed_10m", "pressure_msl",
                "temperature_2m", "wind_direction_100m"):
        d = merged[f"{col}_new"] - merged[f"{col}_old"]
        d = d.dropna()
        if len(d) == 0:
            continue
        print(f"  {col:<25}  mean_diff={d.mean():>+8.4f}  "
              f"std={d.std():>7.4f}  "
              f"max_abs={d.abs().max():>7.4f}")
    # Direction is circular; report wrapped diff.
    dirs = merged.dropna(subset=["wind_direction_100m_new", "wind_direction_100m_old"])
    if len(dirs):
        d = dirs["wind_direction_100m_new"] - dirs["wind_direction_100m_old"]
        d = ((d + 180) % 360) - 180   # wrap to [-180, +180]
        print(f"  wind_dir_100m wrapped     mean_diff={d.mean():>+8.4f}  "
              f"std={d.std():>7.4f}  "
              f"max_abs={d.abs().max():>7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--lat", type=float, default=LAT_REFINED)
    ap.add_argument("--lon", type=float, default=LON_REFINED)
    args = ap.parse_args()

    df = fetch_era5_refined(args.start, args.end, args.lat, args.lon)

    # Match column ordering of the existing parquet for drop-in compat.
    cols = ["time"] + list(HOURLY_VARS)
    df = df[cols]

    # Save.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    sz_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {OUTPUT_PATH} ({sz_mb:.2f} MB, {len(df)} rows)")
    print(f"  Range: {df['time'].min()} → {df['time'].max()}")
    print(f"  Wind 100m stats: mean={df['wind_speed_100m'].mean():.2f} m/s, "
          f"max={df['wind_speed_100m'].max():.2f} m/s")

    compare_to_existing(df)


if __name__ == "__main__":
    main()
