"""Check wind_direction units and 180m gap distribution."""
from __future__ import annotations

import pandas as pd

df = pd.read_csv(r"f:\Claude\win_d\data\raw\train_dataset.csv")
for c in ["wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m"]:
    s = df[c].dropna()
    print(f"{c}: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f} n={len(s)}")
for c in ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m", "wind_gusts_10m"]:
    s = df[c].dropna()
    print(f"{c}: min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}")

# Where are 180m NaN?
df["ts"] = pd.to_datetime(df["METEOFORECASTHOUR_OPENM_Datetime"])
nan_mask = df["wind_speed_180m"].isna()
print("\n180m NaN ts min/max:", df.loc[nan_mask, "ts"].min(), df.loc[nan_mask, "ts"].max())
print("180m NaN count by year:")
print(df.loc[nan_mask, "ts"].dt.year.value_counts().sort_index())

# Valid check: any NaN at 180m?
v = pd.read_csv(r"f:\Claude\win_d\data\raw\valid_features.csv")
print("\nVALID 180m NaN:", v["wind_speed_180m"].isna().sum())
print("VALID wind_direction_180m min/max:", v["wind_direction_180m"].min(), v["wind_direction_180m"].max())

# Correlations between target and key weather vars
y = df["Выработка. Результирующий расчет"]
for c in ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]:
    s = df[c]
    mask = ~s.isna() & ~y.isna()
    print(f"corr y vs {c}: {y[mask].corr(s[mask]):.4f}")
