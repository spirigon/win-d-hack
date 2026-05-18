"""Fetch AI weather model forecasts from Open-Meteo Historical Forecast API.

Models:
- ECMWF AIFS (0.25°) — AI-based, operational since Feb 2025
- GFS GraphCast (0.25°) — Google DeepMind, operational since 2024

Variables: wind_speed_100m, wind_speed_10m, wind_gusts_10m, pressure_msl,
           temperature_2m, wind_direction_100m, wind_direction_10m

Coverage: 2022-01-01 to 2026-03-31 (same as ERA5 cache)

Output: data/external/ai_weather_models.parquet

Usage:
    python scripts/fetch_ai_weather.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAT = 46.8268
LON = 38.7179
OUTPUT = ROOT / "data" / "external" / "ai_weather_models.parquet"

# Variables to fetch (same as ERA5 for comparability).
HOURLY_VARS = [
    "wind_speed_100m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "temperature_2m",
    "wind_direction_100m",
    "wind_direction_10m",
]

MODELS = ["ecmwf_aifs025_single", "gfs_graphcast025"]
MODEL_SHORT = {"ecmwf_aifs025_single": "aifs", "gfs_graphcast025": "graphcast"}

# Fetch in 3-month chunks to avoid API limits.
DATE_RANGES = []
for year in range(2022, 2027):
    for q_start, q_end in [("01-01", "03-31"), ("04-01", "06-30"),
                            ("07-01", "09-30"), ("10-01", "12-31")]:
        start = f"{year}-{q_start}"
        end = f"{year}-{q_end}"
        if year == 2026 and q_start == "04-01":
            break
        DATE_RANGES.append((start, end))


def fetch_model(model: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch one model for one date range."""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(HOURLY_VARS),
        "models": model,
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                hourly = data.get("hourly", {})
                if "time" not in hourly:
                    return None
                df = pd.DataFrame(hourly)
                df["time"] = pd.to_datetime(df["time"])
                return df
            elif r.status_code == 429:
                time.sleep(10)
            else:
                print(f"    Error {r.status_code}: {r.text[:200]}")
                return None
        except Exception as e:
            print(f"    Exception: {e}")
            time.sleep(5)
    return None


def main():
    print("Fetching AI weather model data...")
    print(f"  Models: {MODELS}")
    print(f"  Date ranges: {len(DATE_RANGES)} quarters")
    print(f"  Variables: {HOURLY_VARS}")

    all_dfs = {}
    for model in MODELS:
        short = MODEL_SHORT[model]
        print(f"\n--- {model} ({short}) ---")
        model_dfs = []
        for start, end in DATE_RANGES:
            print(f"  {start} -> {end}...", end=" ")
            df = fetch_model(model, start, end)
            if df is not None:
                model_dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("FAILED")
            time.sleep(1)  # Rate limit courtesy.

        if model_dfs:
            combined = pd.concat(model_dfs, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
            # Rename columns with model prefix.
            rename = {v: f"{short}_{v}" for v in HOURLY_VARS}
            combined = combined.rename(columns=rename)
            all_dfs[short] = combined
            print(f"  Total: {len(combined)} rows")

    # Merge all models on time.
    if not all_dfs:
        print("No data fetched!")
        return

    result = None
    for short, df in all_dfs.items():
        if result is None:
            result = df
        else:
            result = result.merge(df[["time"] + [c for c in df.columns if c != "time"]],
                                  on="time", how="outer")

    result = result.sort_values("time").reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT} ({len(result)} rows, {len(result.columns)} columns)")
    print(f"Columns: {list(result.columns)}")
    print(f"Date range: {result['time'].min()} to {result['time'].max()}")
    # Check NaN rates.
    for c in result.columns:
        if c != "time":
            nan_pct = result[c].isna().mean() * 100
            if nan_pct > 5:
                print(f"  WARNING: {c} has {nan_pct:.1f}% NaN")


if __name__ == "__main__":
    main()
