"""Fetch ECMWF-IFS archive wind data at farm coordinates.

ECMWF IFS reanalysis via Open-Meteo archive API. Provides an independent
NWP estimate that has ~0.96 correlation with ERA5 wind at 100m, giving
useful ensemble diversity (1.26 m/s std difference).

Available from 2022-01-01 (full training period coverage).

Variables:
    wind_speed_100m      — hub-height wind (main signal)
    wind_direction_100m  — direction at 100m
    wind_speed_10m       — surface wind (for shear with 100m)
    wind_gusts_10m       — peak gust (different physics from ERA5)
    temperature_2m       — near-surface temperature
    pressure_msl         — surface pressure

Output: data/external/ecmwf_ifs.parquet

Usage:
    python scripts/fetch_ecmwf_ifs.py
    python scripts/fetch_ecmwf_ifs.py --start 2022-01-01 --end 2026-03-31
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

HOURLY_VARS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
]

OUTPUT = _ROOT / "data" / "external" / "ecmwf_ifs.parquet"


def _fetch_chunk(start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "models": "ecmwf_ifs",
        "timezone": "UTC",
    }
    print(f"  GET archive-api.open-meteo.com (ecmwf_ifs, {start}..{end})...")
    r = requests.get(BASE_URL, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js:
        raise RuntimeError(f"No 'hourly' key in response: {list(js.keys())}")
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_all(start: str, end: str) -> pd.DataFrame:
    year_chunks = [
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", "2026-12-31"),
    ]
    # Clip to actual start/end
    chunks = [(max(s, start), min(e, end)) for s, e in year_chunks if s <= end and e >= start]
    chunks = [(max(s, start), min(e, end)) for s, e in chunks]
    chunks = [(s, e) for s, e in chunks if s <= e]

    dfs = []
    for s, e in chunks:
        t0 = time.time()
        try:
            df = _fetch_chunk(s, e)
            valid_pct = df["wind_speed_100m"].notna().mean()
            print(f"    {len(df)} rows in {time.time() - t0:.1f}s  (ws100m: {valid_pct:.0%} valid)")
            dfs.append(df)
        except Exception as exc:
            print(f"    ERROR: {exc}")
        time.sleep(0.5)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-03-31")
    args = ap.parse_args()

    print(f"Fetching ECMWF-IFS at ({LAT:.6f}N, {LON:.6f}E)")
    print(f"  Period: {args.start} to {args.end}")
    t0 = time.time()
    df = fetch_all(args.start, args.end)
    print(f"\nTotal: {len(df)} rows in {time.time() - t0:.1f}s")
    print(f"  Range: {df['time'].min()} to {df['time'].max()}")
    print(f"  ws_100m:  mean={df['wind_speed_100m'].mean():.2f}  "
          f"max={df['wind_speed_100m'].max():.2f}  "
          f"NaN={df['wind_speed_100m'].isna().sum()}")
    print(f"  dir_100m: mean={df['wind_direction_100m'].mean():.1f} deg")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"  Saved: {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.2f} MB)")

    # Compare against ERA5 if available
    era5_path = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
    if era5_path.exists():
        import numpy as np
        era5 = pd.read_parquet(era5_path)
        era5["time"] = pd.to_datetime(era5["time"])
        m = pd.merge(
            df[["time", "wind_speed_100m", "wind_direction_100m"]],
            era5[["time", "wind_speed_100m", "wind_direction_100m"]],
            on="time", suffixes=("_ecmwf", "_era5"),
        ).dropna()
        corr = np.corrcoef(m["wind_speed_100m_ecmwf"], m["wind_speed_100m_era5"])[0, 1]
        diff = m["wind_speed_100m_ecmwf"] - m["wind_speed_100m_era5"]
        print(f"\nECMWF vs ERA5 wind_speed_100m ({len(m)} matched rows):")
        print(f"  corr={corr:.4f}  mean_diff={diff.mean():+.4f}  std_diff={diff.std():.4f}")


if __name__ == "__main__":
    main()
