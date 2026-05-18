"""Check if IFS data is essentially the same as our training NWP data."""
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
ifs = pd.read_parquet(_ROOT / "data" / "external" / "ecmwf_ifs.parquet")
train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])

merged = ifs.copy().rename(columns={"time": "ts"})
merged = merged.merge(
    train[["ts", "wind_speed_10m", "wind_speed_80m", "wind_gusts_10m", "pressure_msl"]],
    on="ts", suffixes=("_ifs", "_train"),
)
print(f"Merged rows: {len(merged)}")

for col in ["wind_speed_10m", "wind_gusts_10m", "pressure_msl"]:
    train_col = col + "_train"
    ifs_col = col + "_ifs"
    if train_col in merged.columns and ifs_col in merged.columns:
        diff = merged[train_col] - merged[ifs_col]
        corr = merged[train_col].corr(merged[ifs_col])
        print(f"\n{col}:")
        print(f"  Train mean: {merged[train_col].mean():.3f}, IFS mean: {merged[ifs_col].mean():.3f}")
        print(f"  Correlation: {corr:.6f}")
        print(f"  Mean abs diff: {diff.abs().mean():.3f}")
        print(f"  Max abs diff: {diff.abs().max():.3f}")

# Compare IFS 100m vs train 80m.
diff = merged["wind_speed_80m"] - merged["wind_speed_100m"]
corr = merged["wind_speed_80m"].corr(merged["wind_speed_100m"])
print(f"\nTrain 80m vs IFS 100m (expected ~0.3 m/s shear):")
print(f"  Correlation: {corr:.6f}")
print(f"  Mean diff (80m - 100m): {diff.mean():.3f}")

# Compare IFS with ERA5.
era5 = pd.read_parquet(_ROOT / "data" / "external" / "era5_reanalysis.parquet")
merged2 = ifs.copy().rename(columns={"time": "ts"})
merged2 = merged2.merge(era5.rename(columns={"time": "ts"}), on="ts", suffixes=("_ifs", "_era5"))
print(f"\n\nIFS vs ERA5:")
for col in ["wind_speed_10m", "wind_speed_100m", "pressure_msl"]:
    ifs_col = col + "_ifs"
    era5_col = col + "_era5"
    if ifs_col in merged2.columns and era5_col in merged2.columns:
        corr = merged2[ifs_col].corr(merged2[era5_col])
        diff = merged2[ifs_col] - merged2[era5_col]
        print(f"  {col}: corr={corr:.4f}, mean_diff={diff.mean():.3f}")
