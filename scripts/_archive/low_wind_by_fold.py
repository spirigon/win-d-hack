"""Check low-wind power by fold to understand Q1 2025 vs Q1 2026 differences."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.features.physics import compute_air_density, compute_v_eff
from src.data.schema import TARGET_COL

train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
train["density"] = compute_air_density(train["pressure_msl"], train["temperature_80m"])
train["v_eff"] = compute_v_eff(train["wind_speed_120m"], train["density"])
train["year"] = train["ts"].dt.year
train["month"] = train["ts"].dt.month

# Filter to Q1 data each year, compare low-wind patterns.
q1 = train[train["month"].isin([1, 2, 3])]

print("Low-wind (v_eff < 3) power stats per year Q1:")
for year in sorted(q1["year"].unique()):
    sub = q1[(q1["year"] == year) & (q1["v_eff"] < 3.0)]
    if len(sub) > 0:
        print(f"  Q1 {year}: n={len(sub):4d}, mean={sub[TARGET_COL].mean():.2f}, median={sub[TARGET_COL].median():.2f}")

# Also by v_eff bin.
print("\nPower distribution in very low wind (v_eff in [2, 3]):")
for year in sorted(q1["year"].unique()):
    sub = q1[(q1["year"] == year) & (q1["v_eff"] >= 2) & (q1["v_eff"] < 3)]
    if len(sub) > 0:
        print(f"  Q1 {year}: n={len(sub):4d}, mean={sub[TARGET_COL].mean():.2f}, median={sub[TARGET_COL].median():.2f}, frac_zero={(sub[TARGET_COL] < 0.5).mean():.3f}")

# Valid set v_eff distribution.
valid = pd.read_csv(_ROOT / "data" / "raw" / "valid_features.csv")
valid["density"] = compute_air_density(valid["pressure_msl"], valid["temperature_80m"])
valid["v_eff"] = compute_v_eff(valid["wind_speed_120m"], valid["density"])
low_wind_valid = (valid["v_eff"] < 3.0).sum()
print(f"\nValid Q1 2026: {low_wind_valid} rows with v_eff<3 out of {len(valid)} ({low_wind_valid/len(valid)*100:.1f}%)")
