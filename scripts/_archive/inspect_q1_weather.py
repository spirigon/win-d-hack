"""Check how often icing/freezing conditions appear in Q1 2026 valid set.

If temperatures rarely go below 0 or precipitation is rare during cold hours,
an icing derate won't trigger and is a waste of effort.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def main():
    valid = pd.read_csv(ROOT / "data" / "raw" / "valid_features.csv")
    train = pd.read_csv(ROOT / "data" / "raw" / "train_dataset.csv")

    print("Q1 2026 valid set (2126 rows):")
    print("-" * 60)
    print(f"  Temperature 80m: mean={valid['temperature_80m'].mean():.2f}, "
          f"min={valid['temperature_80m'].min():.2f}, max={valid['temperature_80m'].max():.2f}")
    print(f"  Below 0°C: {(valid['temperature_80m'] < 0).sum()} rows "
          f"({(valid['temperature_80m'] < 0).mean()*100:.1f}%)")
    print(f"  Below 2°C: {(valid['temperature_80m'] < 2).sum()} rows "
          f"({(valid['temperature_80m'] < 2).mean()*100:.1f}%)")
    print(f"  Any precip (rain+shower+snow > 0): {((valid['rain']+valid['showers']+valid['snowfall']) > 0).sum()} rows "
          f"({((valid['rain']+valid['showers']+valid['snowfall']) > 0).mean()*100:.1f}%)")
    print(f"  Snow > 0: {(valid['snowfall'] > 0).sum()} rows")
    print(f"  Classic icing (T<0 AND precip>0): {((valid['temperature_80m'] < 0) & ((valid['rain']+valid['showers']+valid['snowfall']) > 0)).sum()} rows")
    print(f"  Rime ice risk (T<2 AND precip>0): {((valid['temperature_80m'] < 2) & ((valid['rain']+valid['showers']+valid['snowfall']) > 0)).sum()} rows")

    # Compare to Q1 training periods.
    train["_ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    for year in [2023, 2024, 2025]:
        q1 = train[(train["_ts"].dt.year == year) & (train["_ts"].dt.month.isin([1, 2, 3]))]
        if len(q1) == 0:
            continue
        below_0 = (q1["temperature_80m"] < 0).sum()
        below_2 = (q1["temperature_80m"] < 2).sum()
        icing_classic = ((q1["temperature_80m"] < 0) & ((q1["rain"] + q1["showers"] + q1["snowfall"]) > 0)).sum()
        icing_rime = ((q1["temperature_80m"] < 2) & ((q1["rain"] + q1["showers"] + q1["snowfall"]) > 0)).sum()
        print(f"\nQ1 {year} training ({len(q1)} rows):")
        print(f"  T<0: {below_0} ({below_0/len(q1)*100:.1f}%)")
        print(f"  T<2: {below_2} ({below_2/len(q1)*100:.1f}%)")
        print(f"  T<0 & precip: {icing_classic} ({icing_classic/len(q1)*100:.1f}%)")
        print(f"  T<2 & precip: {icing_rime} ({icing_rime/len(q1)*100:.1f}%)")


if __name__ == "__main__":
    main()
