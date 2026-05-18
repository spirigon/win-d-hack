"""Where do v3.1 and v8.0 disagree most?"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.features.physics import compute_air_density

valid = pd.read_csv(_ROOT / "data" / "raw" / "valid_features.csv")
v31 = pd.read_csv(_ROOT / "submissions" / "archive" / "v3.1_era5_blend.csv", header=None)[0].to_numpy()
v80 = pd.read_csv(_ROOT / "submissions" / "archive" / "v8.0_wake.csv", header=None)[0].to_numpy()

valid["v31"] = v31
valid["v80"] = v80
valid["diff"] = v80 - v31
valid["density"] = compute_air_density(valid["pressure_msl"], valid["temperature_80m"])

# Most disagreement.
print("Top 15 disagreement rows (v8 - v3.1):")
worst = valid.reindex(valid["diff"].abs().sort_values(ascending=False).index).head(15)
print(worst[["METEOFORECASTHOUR_OPENM_Datetime", "wind_speed_80m", "wind_speed_120m", "wind_direction_120m", "v31", "v80", "diff"]].to_string(index=False))

# By wind speed bin.
valid["ws_bin"] = pd.cut(valid["wind_speed_120m"], [0, 3, 5, 7, 10, 14, 18, 25])
print("\nMean prediction by wind speed bin:")
stats = valid.groupby("ws_bin", observed=True).agg(
    n=("diff", "size"),
    v31_mean=("v31", "mean"),
    v80_mean=("v80", "mean"),
    mean_diff=("diff", "mean"),
    abs_diff=("diff", lambda x: np.mean(np.abs(x))),
).round(3)
print(stats)

# By direction sector.
valid["dir_sector"] = ((valid["wind_direction_120m"] * 1000) // 22.5).astype(int) % 16
print("\nMean prediction by direction sector (16):")
stats_dir = valid.groupby("dir_sector").agg(
    n=("diff", "size"),
    v31_mean=("v31", "mean"),
    v80_mean=("v80", "mean"),
    mean_diff=("diff", "mean"),
).round(3)
print(stats_dir)

# v8 has higher mean — let's see why.
print(f"\nv8 mean - v3.1 mean: {valid['diff'].mean():.3f} MW")
print(f"v8 systematically predicts higher by {valid['diff'].mean():.2f} MW on average")
