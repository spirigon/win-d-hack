"""Feature engineering for the wind farm tabular pipeline.

Design choices:
- Wind direction in the raw CSVs is encoded as degrees / 1000 (range 0.001..0.360),
  decoded here to radians so sin/cos are physical.
- Multi-level wind speed enables REWS (Rotor-Equivalent Wind Speed) across the
  rotor swept area.
- Physics-informed features per IEC 61400-12-1: air density correction, REWS,
  Hellmann shear exponent, density-corrected v_eff.
- Hub height 120 m ≈ closest to actual hub; use as primary wind speed reference.

Research backing: see `Wind Power Forecasting ML Research` doc — r=0.837 at
120m, highest correlation of any raw feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import (
    CLOUD_COL,
    GUST_COL,
    HUB_HEIGHT_M,
    PRECIP_COLS,
    PRESSURE_COL,
    TARGET_COL,
    TIMESTAMP_COL,
    TOTAL_TURBINES,
    TURBINES_IN_MAINTENANCE_COL,
    WD_COLS,
    WS_COLS,
)
from src.data.outliers import add_imputation_flag
from src.features.physics import (
    compute_air_density,
    compute_hellmann_alpha,
    compute_rews,
    compute_v_eff,
    compute_wpd,
)

# Degrees per raw-direction unit. Raw max == 0.360 -> 360 degrees.
DEG_PER_UNIT: float = 1000.0


def _direction_to_radians(series: pd.Series) -> pd.Series:
    """Raw direction (0.001..0.360) -> radians."""
    return np.deg2rad(series.astype(float) * DEG_PER_UNIT)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TIMESTAMP_COL]
    hour = ts.dt.hour
    doy = ts.dt.dayofyear
    month = ts.dt.month
    dow = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["is_weekend"] = (dow >= 5).astype(np.int8)
    df["year_index"] = (ts.dt.year - 2022).astype(np.int16)
    # Seasonal flags per research doc.
    df["is_winter"] = ts.dt.month.isin([12, 1, 2]).astype(np.int8)
    df["is_q1"] = ts.dt.month.isin([1, 2, 3]).astype(np.int8)
    return df


def _impute_180m(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing 180m wind using Hellmann power law from 120m.

    The training file's 180 m levels are NaN for 2022-01-01..2022-11-16 (about
    6 800 rows). Dropping those rows would shrink training by ~20 %. Using
    Hellmann power law from 120m with global mean α=0.233 gives a physically
    consistent estimate.

    v_180 = v_120 × (180/120)^α
    """
    df = df.copy()
    # Compute α where both levels are available.
    ws120 = df["wind_speed_120m"]
    ws180 = df["wind_speed_180m"]

    # Mean α from non-missing rows.
    valid = ws180.notna() & (ws120 > 0.1) & (ws180 > 0.1)
    if valid.sum() > 0:
        alpha_mean = float(
            np.log(ws180[valid] / ws120[valid]).mean() / np.log(180.0 / 120.0)
        )
    else:
        alpha_mean = 0.143  # 1/7 law fallback

    # Impute.
    imputed_180 = ws120 * (180.0 / 120.0) ** alpha_mean
    df["wind_speed_180m"] = ws180.fillna(imputed_180)

    # For direction, fall back to 120m (directions are highly correlated across heights).
    df["wind_direction_180m"] = df["wind_direction_180m"].fillna(df["wind_direction_120m"])

    return df


def _add_wind(df: pd.DataFrame) -> pd.DataFrame:
    # Decode direction to sin/cos for every vertical level.
    for col in WD_COLS:
        rad = _direction_to_radians(df[col])
        level = col.split("_")[-1]
        df[f"wind_dir_{level}_sin"] = np.sin(rad)
        df[f"wind_dir_{level}_cos"] = np.cos(rad)

    # 80 m hub-height proxy from the 10 m level (1/7 power law).
    df["ws_hub_proxy"] = df["wind_speed_10m"] * (HUB_HEIGHT_M / 10.0) ** (1.0 / 7.0)

    # Cube of wind speed at multiple levels (dominant term in wind power).
    for col in WS_COLS:
        df[f"{col}_cube"] = df[col] ** 3
    df["ws_80m_squared"] = df["wind_speed_80m"] ** 2
    df["ws_120m_squared"] = df["wind_speed_120m"] ** 2

    # Vertical shear: speed difference across levels.
    df["ws_shear_10_80"] = df["wind_speed_80m"] - df["wind_speed_10m"]
    df["ws_shear_80_120"] = df["wind_speed_120m"] - df["wind_speed_80m"]
    df["ws_shear_120_180"] = df["wind_speed_180m"] - df["wind_speed_120m"]
    df["ws_shear_ratio_80_120"] = df["wind_speed_120m"] / (df["wind_speed_80m"] + 1e-3)

    # Gust ratio at 10 m (turbulence intensity surrogate).
    df["gust_ratio_10m"] = df[GUST_COL] / (df["wind_speed_10m"] + 1e-3)
    df["gust_excess_10m"] = (df[GUST_COL] - df["wind_speed_10m"]).clip(lower=0.0)

    # Directional veer: angle change between levels (via sin/cos dot product).
    for low, high in (("10m", "80m"), ("80m", "120m"), ("120m", "180m")):
        dot = (
            df[f"wind_dir_{low}_sin"] * df[f"wind_dir_{high}_sin"]
            + df[f"wind_dir_{low}_cos"] * df[f"wind_dir_{high}_cos"]
        )
        df[f"veer_{low}_{high}"] = np.arccos(dot.clip(-1.0, 1.0))

    # Signed veering (positive = clockwise with height = warm advection/stable,
    # negative = backing = cold advection/unstable). Uses cross product for sign.
    for low, high in (("10m", "80m"), ("80m", "120m"), ("120m", "180m"), ("10m", "120m")):
        cross = (
            df[f"wind_dir_{low}_sin"] * df[f"wind_dir_{high}_cos"]
            - df[f"wind_dir_{low}_cos"] * df[f"wind_dir_{high}_sin"]
        )
        dot = (
            df[f"wind_dir_{low}_sin"] * df[f"wind_dir_{high}_sin"]
            + df[f"wind_dir_{low}_cos"] * df[f"wind_dir_{high}_cos"]
        )
        df[f"veer_signed_{low}_{high}"] = np.arctan2(cross, dot)

    # REWS — Rotor-Equivalent Wind Speed across 80/120/180m.
    df["rews"] = compute_rews(
        df["wind_speed_80m"], df["wind_speed_120m"], df["wind_speed_180m"]
    )
    df["rews_cube"] = df["rews"] ** 3

    # Hellmann shear exponent (atmospheric stability).
    df["hellmann_alpha"] = compute_hellmann_alpha(df["wind_speed_10m"], df["wind_speed_120m"])

    return df


def _add_atmos(df: pd.DataFrame) -> pd.DataFrame:
    # Air density at hub height.
    df["air_density"] = compute_air_density(df[PRESSURE_COL], df["temperature_80m"])
    df["density_ratio"] = df["air_density"] / 1.225

    # Density-corrected effective wind speed at hub (120m).
    df["v_eff"] = compute_v_eff(df["wind_speed_120m"], df["air_density"])
    df["v_eff_cube"] = df["v_eff"] ** 3

    # Wind power density.
    df["wpd_120m"] = compute_wpd(df["wind_speed_120m"], df["air_density"])

    # Density-corrected cubed signals.
    df["ws_80m_cube_density"] = df["wind_speed_80m"] ** 3 * df["density_ratio"]
    df["ws_120m_cube_density"] = df["wind_speed_120m"] ** 3 * df["density_ratio"]

    df["temp_gradient"] = df["temperature_120m"] - df["temperature_80m"]
    df["precip_total"] = sum(df[c] for c in PRECIP_COLS)
    df["has_precip"] = (df["precip_total"] > 0).astype(np.int8)
    # Icing risk indicator (freezing + precip).
    df["icing_risk"] = ((df["temperature_80m"] < 2) & (df["precip_total"] > 0)).astype(np.int8)

    # Enhanced icing features (Q1 winter-specific).
    # Continuous icing severity: colder + more precip = worse icing.
    freezing_margin = (2.0 - df["temperature_80m"]).clip(lower=0.0)  # degrees below 2°C
    df["icing_severity"] = freezing_margin * df["precip_total"]
    # Icing × wind speed interaction: icing at high wind = more blade accretion.
    df["icing_x_wind"] = df["icing_severity"] * df["wind_speed_120m"]
    # Freezing rain (most dangerous for icing): rain when T < 0.
    df["freezing_rain_risk"] = (df["rain"] * (-df["temperature_80m"]).clip(lower=0.0))
    # Snow accumulation risk (lighter icing but still relevant).
    df["snow_accum_risk"] = df["snowfall"] * freezing_margin
    # Cloud cover × freezing (radiative cooling + moisture = rime ice).
    df["rime_ice_proxy"] = (
        (df[CLOUD_COL] / 100.0) * freezing_margin * (df["wind_speed_80m"] > 3).astype(np.float32)
    )

    return df


def _add_static(df: pd.DataFrame) -> pd.DataFrame:
    df["active_turbines"] = TOTAL_TURBINES - df[TURBINES_IN_MAINTENANCE_COL]
    df["active_turbines_ratio"] = df["active_turbines"] / TOTAL_TURBINES
    df["maintenance_ratio"] = df[TURBINES_IN_MAINTENANCE_COL] / TOTAL_TURBINES
    # Interactions.
    df["rews_cube_x_active"] = df["rews"] ** 3 * df["active_turbines_ratio"]
    df["v_eff_cube_x_active"] = df["v_eff"] ** 3 * df["active_turbines_ratio"]
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features: direction × speed, hour × speed, etc."""
    # Direction sector (8-way) × wind speed.
    dir_deg = df["wind_direction_120m"] * DEG_PER_UNIT
    df["dir_sector_8"] = (dir_deg // 45).clip(0, 7).fillna(0).astype(np.int8)
    df["dir_sector_16"] = (dir_deg // 22.5).clip(0, 15).fillna(0).astype(np.int8)
    df["rews_x_sector8"] = df["rews"] * df["dir_sector_8"]

    # ws × hour (diurnal stability).
    hour = df[TIMESTAMP_COL].dt.hour
    df["rews_x_hour_sin"] = df["rews"] * np.sin(2 * np.pi * hour / 24.0)
    df["rews_x_hour_cos"] = df["rews"] * np.cos(2 * np.pi * hour / 24.0)

    # Shear × gust (turbulence signal).
    df["shear_x_gust"] = df["ws_shear_10_80"] * df[GUST_COL]

    # Density-corrected × active turbines.
    df["wpd_x_active"] = df["wpd_120m"] * df["active_turbines_ratio"]

    return df


def build_features(df: pd.DataFrame, sort_by_time: bool = True) -> pd.DataFrame:
    """Apply the full feature pipeline to a training or validation frame."""
    out = df.copy()
    if sort_by_time:
        out = out.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # Capture the pre-imputation state of wind_speed_180m so that
    # add_imputation_flag can identify which rows were NaN-then-filled by
    # the Hellmann power law below (Requirement 3.1–3.4).
    if "wind_speed_180m" in out.columns:
        _pre_impute = out.copy()
        out = _impute_180m(out)
        out = add_imputation_flag(_pre_impute, out, col="wind_speed_180m")
    else:
        out = _impute_180m(out)

    out = _add_calendar(out)
    out = _add_wind(out)
    out = _add_atmos(out)
    out = _add_static(out)
    out = _add_interactions(out)
    # Defragment after many column insertions.
    out = out.copy()
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of columns to feed the model.

    Exclusions: timestamp, target, bookkeeping columns.
    """
    drop = {TIMESTAMP_COL, TARGET_COL, "_submission_row", "_source"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
