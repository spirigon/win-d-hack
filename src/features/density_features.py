"""Air density correction features for wind power forecasting.

Physical motivation:
  Wind turbine power  P = ½ρAv³Cp
  Standard power curves assume ρ₀ = 1.225 kg/m³ (ISO atmosphere: 15°C, 1013.25 hPa).
  Actual density varies ±5% seasonally (winter dense → more power; summer thin → less).
  Below rated wind speed P scales linearly with ρ.
  Above rated the pitch system limits P so density effect is smaller but still real.

Expected features added (~9):
  air_density            : ρ = sp / (R_dry × T)  [kg/m³]
  density_ratio          : ρ / ρ₀
  wind_speed_120m_rho_corrected : v × (ρ/ρ₀)^(1/3)  — iso-power wind speed at std density
  v_eff_rho_corrected    : same but for effective wind speed
  kinetic_power_120m     : ½ρv³ at 120m  — actual available kinetic power density
  kinetic_power_veff     : ½ρv_eff³
  ds_120m_farm_mw_rho    : ds_120m_farm_mw × density_ratio  — density-corrected theoretical power
  ds_85m_pl_farm_mw_rho  : ds_85m_pl_farm_mw × density_ratio
  t2m_vs_iso             : T - 288.15 K  — temperature anomaly from ISO standard
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RHO_STD = 1.225      # kg/m³  ISO standard air density
R_DRY   = 287.058    # J/(kg·K) specific gas constant for dry air


def add_density_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sp" not in df.columns or "t2m" not in df.columns:
        return df

    sp  = df["sp"].to_numpy(dtype=np.float64)
    t2m = df["t2m"].to_numpy(dtype=np.float64)

    # Unit guards: ERA5 sp is Pa (~101325), t2m is K (~288)
    # If medians suggest different units, correct silently
    sp_med  = float(np.nanmedian(sp))
    t2m_med = float(np.nanmedian(t2m))
    if sp_med < 2_000:            # looks like hPa
        sp = sp * 100.0
    if t2m_med < 100:             # looks like °C
        t2m = t2m + 273.15

    rho       = sp / (R_DRY * t2m)           # kg/m³
    rho_ratio = rho / RHO_STD                 # dimensionless

    df["air_density"]   = rho.astype(np.float32)
    df["density_ratio"] = rho_ratio.astype(np.float32)

    # ISO-power corrected wind speed: same kinetic power at ρ₀
    # ½ρ₀v_std³ = ½ρv_actual³  →  v_std = v_actual × (ρ/ρ₀)^(1/3)
    for col in ("wind_speed_120m", "v_eff"):
        if col in df.columns:
            df[f"{col}_rho_corrected"] = (
                df[col].to_numpy(dtype=np.float64) * rho_ratio ** (1.0 / 3.0)
            ).astype(np.float32)

    # Actual kinetic power density [W/m²] — proportional to available power
    for col in ("wind_speed_120m", "v_eff"):
        label = "120m" if col == "wind_speed_120m" else "veff"
        if col in df.columns:
            v = df[col].to_numpy(dtype=np.float64)
            df[f"kinetic_power_{label}"] = (0.5 * rho * v**3).astype(np.float32)

    # Density-corrected datasheet power (theoretical × actual/std density)
    for ds_col in ("ds_120m_farm_mw", "ds_85m_pl_farm_mw"):
        if ds_col in df.columns:
            df[f"{ds_col}_rho"] = (
                df[ds_col].to_numpy(dtype=np.float64) * rho_ratio
            ).astype(np.float32)

    # Temperature anomaly from ISO standard (proxy for density anomaly)
    df["t2m_vs_iso"] = (t2m - 288.15).astype(np.float32)

    return df


def density_columns(df: pd.DataFrame) -> list[str]:
    targets = {
        "air_density", "density_ratio",
        "wind_speed_120m_rho_corrected", "v_eff_rho_corrected",
        "kinetic_power_120m", "kinetic_power_veff",
        "ds_120m_farm_mw_rho", "ds_85m_pl_farm_mw_rho",
        "t2m_vs_iso",
    }
    return [c for c in df.columns if c in targets]
