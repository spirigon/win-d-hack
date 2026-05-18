"""Compare all NWP sources: training vs ERA5 vs GFS vs ICON vs GEM."""
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
train_sub = train[["ts", "wind_speed_10m", "wind_speed_80m", "wind_gusts_10m", "pressure_msl"]]

era5 = pd.read_parquet(_ROOT / "data" / "external" / "era5_reanalysis.parquet").rename(columns={"time": "ts"})
gfs = pd.read_parquet(_ROOT / "data" / "external" / "gfs_global.parquet").rename(columns={"time": "ts"})
icon = pd.read_parquet(_ROOT / "data" / "external" / "icon_global.parquet").rename(columns={"time": "ts"})

print("Correlations with training wind_speed_80m:")
for name, df in [("ERA5 ws100", era5[["ts", "wind_speed_100m"]]),
                 ("GFS ws100", gfs[["ts", "wind_speed_100m"]]),
                 ("ICON ws100", icon[["ts", "wind_speed_100m"]])]:
    merged = train_sub.merge(df, on="ts", how="inner")
    col = [c for c in df.columns if c != "ts"][0]
    corr = merged["wind_speed_80m"].corr(merged[col])
    mean_diff = (merged["wind_speed_80m"] - merged[col]).mean()
    print(f"  {name}: corr={corr:.4f}, mean_diff={mean_diff:+.3f}, n={len(merged)}")

print("\nCorrelations between different sources' 100m wind speeds:")
m1 = era5[["ts", "wind_speed_100m"]].rename(columns={"wind_speed_100m": "era5"})
m2 = gfs[["ts", "wind_speed_100m"]].rename(columns={"wind_speed_100m": "gfs"})
m3 = icon[["ts", "wind_speed_100m"]].rename(columns={"wind_speed_100m": "icon"})
mm = m1.merge(m2, on="ts", how="inner").merge(m3, on="ts", how="inner").dropna()
print(mm[["era5", "gfs", "icon"]].corr().round(4))

# Correlation with training power.
target = "\u0412\u044b\u0440\u0430\u0431\u043e\u0442\u043a\u0430. \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0440\u0430\u0441\u0447\u0435\u0442"
train_power = train[["ts", target]].rename(columns={target: "power"})
for name, df in [("ERA5 ws100", era5[["ts", "wind_speed_100m"]]),
                 ("GFS ws100", gfs[["ts", "wind_speed_100m"]]),
                 ("ICON ws100", icon[["ts", "wind_speed_100m"]])]:
    merged = train_power.merge(df, on="ts", how="inner").dropna()
    col = [c for c in df.columns if c != "ts"][0]
    corr = merged["power"].corr(merged[col])
    print(f"\nPower vs {name} 100m: corr={corr:.4f}, n={len(merged)}")
