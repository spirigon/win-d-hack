"""Fetch ERA5 reanalysis data from Open-Meteo for our wind farm location.

Coordinates: 46.8268, 38.7179
Period: 2022-01-01 to 2026-03-31
Variables: wind_speed_10m, wind_speed_100m, wind_direction_10m, wind_direction_100m,
           wind_gusts_10m, temperature_2m, pressure_msl, cloud_cover_low, rain, snowfall

Output: data/external/era5_reanalysis.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

LAT = 46.8268
LON = 38.7179
OUTPUT = _ROOT / "data" / "external" / "era5_reanalysis.parquet"

# Open-Meteo Historical Weather API (ERA5 reanalysis).
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
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
]


def fetch_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one date range from the API."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def main():
    # Fetch in yearly chunks to stay within API limits.
    chunks = [
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", "2026-03-31"),
    ]

    all_dfs = []
    for start, end in chunks:
        print(f"  Fetching {start} to {end}...")
        try:
            df = fetch_chunk(start, end)
            all_dfs.append(df)
            print(f"    Got {len(df)} rows")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1)  # Be polite to the API.

    if not all_dfs:
        print("No data fetched!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["time"])
    print(f"\n  Total: {len(combined)} rows, {combined['time'].min()} to {combined['time'].max()}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT, index=False)
    print(f"  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
