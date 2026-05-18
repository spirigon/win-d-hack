"""Fetch multiple NWP models from Open-Meteo for ensemble diversity.

Different NWP models make different assumptions and errors. Having
GFS + ICON + ECMWF + ERA5 gives the LGBM robust multi-source signals.
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

# Historical forecast archive endpoint (available for each NWP model).
BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
]


def fetch_model(model: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
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


def fetch_all_for_model(model: str) -> pd.DataFrame:
    chunks = [
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", "2026-03-31"),
    ]
    all_dfs = []
    for s, e in chunks:
        try:
            df = fetch_model(model, s, e)
            all_dfs.append(df)
            print(f"  {model} {s}-{e}: {len(df)} rows")
        except Exception as exc:
            print(f"  {model} {s}-{e}: ERROR {exc}")
        time.sleep(1)
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"])
    return combined


def main():
    # Try several NWP models.
    models = ["gfs_global", "icon_global", "gem_global"]

    for model in models:
        print(f"\nFetching {model}...")
        df = fetch_all_for_model(model)
        if not df.empty:
            print(f"  Total: {len(df)} rows")
            # Check for data quality.
            valid = df["wind_speed_10m"].notna().sum()
            print(f"  Valid wind_speed_10m: {valid}/{len(df)}")
            out = _ROOT / "data" / "external" / f"{model}.parquet"
            df.to_parquet(out, index=False)
            print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
