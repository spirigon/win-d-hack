"""Verify datasheet power curve interpolation + compare to training data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.features.datasheet_power_curve import (
    POWER_TABLE_KW,
    DENSITY_GRID_SORTED,
    WS_GRID,
    per_turbine_power_kw,
    farm_theoretical_power_mw,
)
from src.features.physics import compute_air_density
from src.data.schema import TARGET_COL, TURBINES_IN_MAINTENANCE_COL, TOTAL_TURBINES

# Test interpolation at known points.
print("Verification at known table entries:")
tests = [
    (5.0, 1.225, 434),   # standard density
    (10.0, 1.225, 3208),
    (15.0, 1.225, 3465),
    (8.0, 1.12, 1823),
    (8.0, 1.27, 2067),
]
for ws, rho, expected in tests:
    got = per_turbine_power_kw(np.array([ws]), np.array([rho]))[0]
    print(f"  ws={ws:4.1f}, rho={rho:.3f}: expected={expected:4.0f}, got={got:4.0f}")

# Compare to training data aggregated.
train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["ts"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
train["air_density"] = compute_air_density(train["pressure_msl"], train["temperature_80m"])
train["active"] = TOTAL_TURBINES - train[TURBINES_IN_MAINTENANCE_COL]

# Theoretical from datasheet using wind_speed_80m.
train["ds_80m_mw"] = farm_theoretical_power_mw(
    train["wind_speed_80m"].to_numpy(),
    train["air_density"].to_numpy(),
    train["active"].to_numpy(),
)
train["ds_120m_mw"] = farm_theoretical_power_mw(
    train["wind_speed_120m"].to_numpy(),
    train["air_density"].to_numpy(),
    train["active"].to_numpy(),
)

# Correlation with observed power.
corr_80 = train[TARGET_COL].corr(train["ds_80m_mw"])
corr_120 = train[TARGET_COL].corr(train["ds_120m_mw"])
print(f"\nCorrelations with actual power:")
print(f"  datasheet(ws_80m): {corr_80:.4f}")
print(f"  datasheet(ws_120m): {corr_120:.4f}")

# Residual stats.
for col, label in (("ds_80m_mw", "80m"), ("ds_120m_mw", "120m")):
    resid = train[TARGET_COL] - train[col]
    print(f"\n  Residuals (actual - datasheet[{label}]):")
    print(f"    mean: {resid.mean():.2f} MW, std: {resid.std():.2f}, MAE: {resid.abs().mean():.2f}")

# Per-wind-speed-bin bias.
print("\n  Bias by wind speed (using ds_120m_mw):")
train["ws_bin"] = pd.cut(train["wind_speed_120m"], [0, 3, 5, 7, 10, 14, 18, 25])
by_bin = train.groupby("ws_bin", observed=True).apply(
    lambda g: pd.Series({
        "n": len(g),
        "mean_actual": g[TARGET_COL].mean(),
        "mean_ds": g["ds_120m_mw"].mean(),
        "mean_bias": (g[TARGET_COL] - g["ds_120m_mw"]).mean(),
    }),
    include_groups=False,
).round(2)
print(by_bin)

# If we used datasheet directly as prediction, what would nMAE be?
mae_datasheet = (train[TARGET_COL] - train["ds_120m_mw"]).abs().mean()
nmae_datasheet = mae_datasheet / 90.09 * 100
print(f"\n  If we just used datasheet(ws_120m) as prediction:")
print(f"    Train nMAE: {nmae_datasheet:.4f} %")

mae_datasheet_80 = (train[TARGET_COL] - train["ds_80m_mw"]).abs().mean()
nmae_datasheet_80 = mae_datasheet_80 / 90.09 * 100
print(f"  If we just used datasheet(ws_80m) as prediction:")
print(f"    Train nMAE: {nmae_datasheet_80:.4f} %")
