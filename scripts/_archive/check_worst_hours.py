"""Investigate the worst errors — are they maintenance/curtailment events?"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
train = train.sort_values("ts").reset_index(drop=True)

TARGET = "\u0412\u044b\u0440\u0430\u0431\u043e\u0442\u043a\u0430. \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0440\u0430\u0441\u0447\u0435\u0442"

# Look at March 11-12 2025 — model predicted ~65 MW but actual was 5-7 MW
# This screams maintenance/curtailment
mask = (train["ts"] >= "2025-03-11") & (train["ts"] <= "2025-03-13")
cols = ["ts", TARGET, "wind_speed_80m", "wind_gusts_10m", "\u041a\u043e\u043b-\u0432\u043e_\u0412\u042d\u0423_\u0432_\u0440\u0435\u043c\u043e\u043d\u0442\u0435"]
print("=== March 11-13 2025 ===")
print(train.loc[mask, cols].to_string(index=False))

# How many hours have ws_80m > 5 but power < 5?
high_ws_low_power = (train["wind_speed_80m"] > 5) & (train[TARGET] < 5)
print(f"\nHours with ws_80m > 5 AND power < 5: {high_ws_low_power.sum()} / {len(train)}")
print(f"  = {high_ws_low_power.mean()*100:.2f}%")

# Maintenance column distribution
print(f"\nMaintenance column distribution:")
print(train["\u041a\u043e\u043b-\u0432\u043e_\u0412\u042d\u0423_\u0432_\u0440\u0435\u043c\u043e\u043d\u0442\u0435"].value_counts().sort_index())

# Check if power is correlated with maintenance count
print("\nMean power by maintenance count:")
print(train.groupby("\u041a\u043e\u043b-\u0432\u043e_\u0412\u042d\u0423_\u0432_\u0440\u0435\u043c\u043e\u043d\u0442\u0435")[TARGET].mean().round(2))

# Lag-1 power autocorrelation
train["power_lag1"] = train[TARGET].shift(1)
print(f"\nPower lag-1 autocorrelation: {train[TARGET].corr(train['power_lag1']):.4f}")
print(f"Power lag-2 autocorrelation: {train[TARGET].corr(train[TARGET].shift(2)):.4f}")
print(f"Power lag-3 autocorrelation: {train[TARGET].corr(train[TARGET].shift(3)):.4f}")
print(f"Power lag-6 autocorrelation: {train[TARGET].corr(train[TARGET].shift(6)):.4f}")
print(f"Power lag-24 autocorrelation: {train[TARGET].corr(train[TARGET].shift(24)):.4f}")

# Weather lag autocorrelation
print(f"\nws_80m lag-1 autocorrelation: {train['wind_speed_80m'].corr(train['wind_speed_80m'].shift(1)):.4f}")
print(f"ws_80m lag-3 autocorrelation: {train['wind_speed_80m'].corr(train['wind_speed_80m'].shift(3)):.4f}")
