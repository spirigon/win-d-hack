"""Fetch CERRA reanalysis data for Europe (5km resolution).

CERRA covers Europe at 5km resolution - much finer than ERA5's 25km.
Our farm (46.83N, 38.72E) is near the Sea of Azov, in the CERRA domain.

Note: CERRA only goes to June 2021, so for 2022+ we still need ERA5/IFS.
Strategy: Not viable as primary, but check ECMWF_IFS instead.
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
OUTPUT = _ROOT / "data" / "external" / "cerra_reanalysis.parquet"
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


def fetch_chunk(start_date: str, end_date: str, model: str = "cerra") -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": model,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def main():
    # CERRA only goes to June 2021 - try ECMWF_IFS instead.
    # ECMWF_IFS from 2017 onward, 9km resolution.
    print("Trying ECMWF IFS (9km)...")
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
            df = fetch_chunk(start, end, model="ecmwf_ifs")
            all_dfs.append(df)
            print(f"    Got {len(df)} rows")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1)

    if not all_dfs:
        print("No data!")
        return
    combined = pd.concat(all_dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["time"])
    print(f"\n  Total: {len(combined)} rows")
    out = _ROOT / "data" / "external" / "ecmwf_ifs.parquet"
    combined.to_parquet(out, index=False)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
