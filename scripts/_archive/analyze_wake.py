"""Analyze wake losses by wind direction sector.

The datasheet gives clean per-turbine power. Farm-level wake losses occur
when downstream turbines are in the wake of upstream ones. Wake loss depends
on wind direction (which determines which turbines are downstream).

We compute: wake_residual = datasheet_farm_mw - actual_power
per (wind_speed_bin, direction_sector).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.features.datasheet_power_curve import farm_theoretical_power_mw
from src.features.physics import compute_air_density
from src.data.schema import TARGET_COL, TURBINES_IN_MAINTENANCE_COL, TOTAL_TURBINES

train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
train["density"] = compute_air_density(train["pressure_msl"], train["temperature_80m"])
train["active"] = TOTAL_TURBINES - train[TURBINES_IN_MAINTENANCE_COL]

train["ds_80m_mw"] = farm_theoretical_power_mw(
    train["wind_speed_80m"].to_numpy(),
    train["density"].to_numpy(),
    train["active"].to_numpy(),
)
train["wake_residual"] = train["ds_80m_mw"] - train[TARGET_COL]
train["dir_deg"] = train["wind_direction_80m"] * 1000.0
train["dir_sector_16"] = (train["dir_deg"] // 22.5).astype(int) % 16

# Exclude obvious curtailment (where ds>>actual by huge margin).
ok_mask = (train["wake_residual"].abs() < 30) & (train[TARGET_COL] > 0.5) & (train["ds_80m_mw"] > 0.5)
train_ok = train[ok_mask].copy()

print(f"Clean rows: {len(train_ok)}")

# Wake residual per direction sector × ws bin.
train_ok["ws_bin"] = pd.cut(train_ok["wind_speed_80m"], [0, 4, 6, 8, 10, 12, 14, 25])

table = train_ok.groupby(["dir_sector_16", "ws_bin"], observed=True)["wake_residual"].agg(["mean", "count"])
print("\nWake residual (ds - actual, MW) by direction sector × ws bin:")
pivot = table["mean"].unstack(level="ws_bin").round(2)
counts = table["count"].unstack(level="ws_bin").round(0)
print(pivot)
print("\nRow counts:")
print(counts)

# Mean wake loss per direction (averaged over ws).
print("\nMean wake loss per direction sector (all ws):")
per_dir = train_ok.groupby("dir_sector_16", observed=True)["wake_residual"].agg(["mean", "std", "count"]).round(2)
print(per_dir)

# Wake loss as fraction of theoretical (relative).
train_ok["wake_frac"] = train_ok["wake_residual"] / train_ok["ds_80m_mw"]
high_ws = train_ok[train_ok["wind_speed_80m"] >= 6]
per_dir_frac = high_ws.groupby("dir_sector_16", observed=True)["wake_frac"].mean().round(3)
print("\nMean wake loss fraction (at ws>=6 m/s):")
print(per_dir_frac)

# What's the overall wake loss?
overall = train_ok["wake_residual"].mean()
print(f"\nOverall mean wake residual: {overall:.2f} MW")
print(f"Overall abs MAE from datasheet: {train_ok['wake_residual'].abs().mean():.2f} MW")
