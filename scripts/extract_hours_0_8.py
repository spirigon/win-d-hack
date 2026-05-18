"""Extract hours 00:00–08:00 from the existing 24-hour forecast.

The hackathon now only requires prediction from 00:00 to 08:00 inclusive
(9 hours). This script filters the existing forecast and writes a clean
submission file.

Usage:
    python scripts/extract_hours_0_8.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW

INPUT   = _ROOT / "submissions" / "18_05_2026_forecast.csv"
OUTPUT  = _ROOT / "submissions" / "18_05_2026_forecast_0_8.csv"


def main():
    df = pd.read_csv(INPUT)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Filter to hours 0-8 inclusive
    mask = df["hour"].between(0, 8)
    out = df[mask].copy().reset_index(drop=True)
    
    print(f"Input:  {len(df)} hours")
    print(f"Output: {len(out)} hours (00:00-08:00)")
    print(f"\nPredictions:")
    print(f"{'Hour':>5}  {'Forecast MW':>12}  {'P10':>8}  {'P90':>8}")
    print(f"  {'-'*45}")
    for _, r in out.iterrows():
        print(f"  {int(r['hour']):>2}h   {r['forecast_mw']:>10.3f}   "
              f"{r['p10_mw']:>7.3f}   {r['p90_mw']:>7.3f}")
    
    print(f"\n  Mean: {out['forecast_mw'].mean():.3f} MW")
    print(f"  Range: [{out['forecast_mw'].min():.3f}, {out['forecast_mw'].max():.3f}] MW")
    
    # Write the submission
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"\n  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
