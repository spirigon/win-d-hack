"""Fetch multi-lead-time historical forecasts from Open-Meteo.

The training data uses OpenMeteo's forecast. We fetch the SAME forecast model
at different lead times to get forecast evolution features.

Open-Meteo Historical Forecast API stores forecasts as they were issued.
For a target hour T, we can get:
- The forecast issued ~24h before (what's in training data)
- The forecast issued ~48h before (older, less accurate)
- The forecast issued ~6h before (fresher, more accurate — but may not exist for valid)

We fetch the "best_match" model (which is what OpenMeteo uses by default)
for the full 2022-2026 period, then compute lead-time disagreement features.

Output: data/external/multi_leadtime_forecasts.parquet

Usage:
    python scripts/fetch_multi_leadtime.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "external" / "multi_leadtime_forecasts.parquet"

LAT = 46.8268
LON = 38.7179

HOURLY_VARS = "wind_speed_80m,wind_speed_120m,wind_speed_10m,wind_gusts_10m,pressure_msl,temperature_80m"

# Fetch in monthly chunks.
def generate_months():
    for year in range(2022, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 3:
                break
            start = f"{year}-{month:02d}-01"
            if month == 12:
                end = f"{year}-12-31"
            else:
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                end = f"{year}-{month:02d}-{last_day:02d}"
            yield start, end


def fetch_chunk(start, end, past_days=2):
    """Fetch historical forecast with past_days parameter.

    past_days=1 gives ~24h lead time forecast
    past_days=2 gives ~48h lead time forecast
    past_days=3 gives ~72h lead time forecast
    """
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": HOURLY_VARS,
        "start_date": start,
        "end_date": end,
        "past_days": past_days,
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
                time.sleep(15)
            else:
                print(f"  Error {r.status_code}: {r.text[:200]}")
                return None
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(5)
    return None


def main():
    print("Fetching multi-lead-time forecasts...")
    print("  This fetches the default OpenMeteo forecast model at different lead times.")

    # We'll fetch with past_days=2 (48h lead) and past_days=3 (72h lead).
    # The training data already has the ~24h lead (past_days=1 equivalent).
    # Comparing 24h vs 48h vs 72h forecasts reveals forecast stability.

    all_dfs = {}
    for lead_label, past_days in [("lead48h", 2), ("lead72h", 3)]:
        print(f"\n--- {lead_label} (past_days={past_days}) ---")
        chunks = []
        for start, end in generate_months():
            print(f"  {start} -> {end}...", end=" ", flush=True)
            df = fetch_chunk(start, end, past_days=past_days)
            if df is not None:
                chunks.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("FAILED")
            time.sleep(0.5)

        if chunks:
            combined = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
            # Rename with lead prefix.
            rename = {v: f"{lead_label}_{v}" for v in HOURLY_VARS.split(",")}
            combined = combined.rename(columns=rename)
            all_dfs[lead_label] = combined
            print(f"  Total: {len(combined)} rows")

    # Merge on time.
    if not all_dfs:
        print("No data!")
        return

    result = None
    for label, df in all_dfs.items():
        if result is None:
            result = df
        else:
            result = result.merge(df[["time"] + [c for c in df.columns if c != "time"]],
                                  on="time", how="outer")

    result = result.sort_values("time").reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT} ({len(result)} rows, {len(result.columns)} cols)")
    print(f"Date range: {result['time'].min()} to {result['time'].max()}")
    # NaN check.
    for c in result.columns:
        if c != "time":
            nan_pct = result[c].isna().mean() * 100
            if nan_pct > 10:
                print(f"  {c}: {nan_pct:.1f}% NaN")


if __name__ == "__main__":
    main()
