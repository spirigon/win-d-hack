"""Fetch ERA5-Land (11 km) at precise farm coordinates.

ERA5-Land runs the ECMWF land-surface scheme at 0.1° (~11 km) vs the
standard ERA5's 0.25° (~28 km). At a coastal site 3 km from the Sea of
Azov the higher resolution means the grid cell better represents the
sea/land contrast. All other API parameters are identical to
``fetch_era5.py``; only ``models`` changes.

Precise farm coordinates: 46.851508°N, 38.708097°E.

Output: data/external/era5_land.parquet

Usage:
    python scripts/fetch_era5_land.py
    python scripts/fetch_era5_land.py --start 2022-01-01 --end 2026-03-31
    python scripts/fetch_era5_land.py --compare   # diff vs era5_reanalysis.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LAT = 46.851508
LON = 38.708097

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ERA5-Land is a surface/near-surface model — pressure-level variables like
# wind_speed_100m are NOT available. We get higher-resolution versions of the
# surface variables (10m wind, gusts, temperature, pressure, precip).
HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
    "cloud_cover_low",
    "rain",
    "snowfall",
]

OUTPUT = _ROOT / "data" / "external" / "era5_land.parquet"


def _fetch_chunk(start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "models": "era5_land",
        "timezone": "UTC",
    }
    print(f"  GET archive-api.open-meteo.com (era5_land, {start}..{end})...")
    r = requests.get(BASE_URL, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js:
        raise RuntimeError(f"No 'hourly' key in response: {list(js.keys())}")
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_all(start: str, end: str) -> pd.DataFrame:
    chunks = [
        (start, "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", end),
    ]
    # Clip first/last chunk to actual start/end.
    chunks[0] = (start, min("2022-12-31", end))
    chunks[-1] = (max("2026-01-01", start), end)
    chunks = [(s, e) for s, e in chunks if s <= e]

    dfs = []
    for s, e in chunks:
        t0 = time.time()
        try:
            df = _fetch_chunk(s, e)
            print(f"    {len(df)} rows in {time.time() - t0:.1f}s")
            dfs.append(df)
        except Exception as exc:
            print(f"    ERROR: {exc}")
        time.sleep(0.5)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return combined


def compare_vs_era5(land_df: pd.DataFrame) -> None:
    era5_path = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
    if not era5_path.exists():
        return
    era5 = pd.read_parquet(era5_path)
    era5["time"] = pd.to_datetime(era5["time"])
    m = pd.merge(
        land_df[["time", "wind_speed_10m", "pressure_msl", "temperature_2m",
                 "wind_direction_10m"]],
        era5[["time", "wind_speed_10m", "pressure_msl", "temperature_2m",
              "wind_direction_10m"]],
        on="time", suffixes=("_land", "_era5"),
    ).dropna()
    print(f"\nDiff stats: ERA5-Land vs ERA5 ({len(m)} matched rows):")
    for col in ("wind_speed_10m", "pressure_msl", "temperature_2m"):
        d = m[f"{col}_land"] - m[f"{col}_era5"]
        print(f"  {col:<28}  mean={d.mean():+.4f}  std={d.std():.4f}  "
              f"max_abs={d.abs().max():.4f}")
    d = m["wind_direction_10m_land"] - m["wind_direction_10m_era5"]
    d = ((d + 180) % 360) - 180
    print(f"  wind_direction_10m           mean={d.mean():+.4f}  std={d.std():.4f}  "
          f"max_abs={d.abs().max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--compare", action="store_true", help="diff vs era5_reanalysis.parquet")
    args = ap.parse_args()

    print(f"Fetching ERA5-Land at ({LAT:.6f}°N, {LON:.6f}°E)")
    print(f"  Period: {args.start} to {args.end}")
    t0 = time.time()
    df = fetch_all(args.start, args.end)
    print(f"\nTotal: {len(df)} rows in {time.time() - t0:.1f}s")
    print(f"  Range: {df['time'].min()} to {df['time'].max()}")
    print(f"  ws_10m:  mean={df['wind_speed_10m'].mean():.2f}  "
          f"max={df['wind_speed_10m'].max():.2f}  "
          f"NaN={df['wind_speed_10m'].isna().sum()}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"  Saved: {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.2f} MB)")

    if args.compare:
        compare_vs_era5(df)


if __name__ == "__main__":
    main()
