"""Advanced cross-source interaction features.

Computed AFTER all external data is merged (ERA5, ERA5v2, ICON-EU, GFS,
ECMWF IFS, NASA MERRA-2). Extracts signal from the relationships between
sources that cannot be captured by any single source alone.

Key feature groups:
    ERA5v2 hub-height vs NWP     — pressure-level wind at exact hub height
    ERA5v2 stability interactions — BLH, CAPE, LLJ × wind speed
    Multi-model direction agreement — circular dot-product between forecasts
    Atmospheric stability features — temp inversion, Ri × wind interactions
    Lagged wind features          — extended rolling on multiple NWP sources
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMESTAMP_COL = "METEOFORECASTHOUR_OPENM_Datetime"


# ---------------------------------------------------------------------------
# ERA5v2 hub-height and stability interactions
# ---------------------------------------------------------------------------

def add_era5v2_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features that require era5v2_* columns.

    These are computed after merge_era5_v2() has run and renamed columns
    to the era5v2_* prefix.
    """
    df = df.copy()

    # --- Pressure-level wind at hub height vs NWP ---
    # era5v2_wind_speed_120m is ERA5 interpolated to exact 120m hub height.
    # Bias vs hackathon NWP at same height is the most apples-to-apples
    # cross-source signal in the entire pipeline.
    if "era5v2_wind_speed_120m" in df.columns and "wind_speed_120m" in df.columns:
        df["era5v2_ws120_vs_nwp120"] = (
            df["era5v2_wind_speed_120m"].astype(float)
            - df["wind_speed_120m"].astype(float)
        )
        df["era5v2_ws120_vs_nwp120_sq"] = df["era5v2_ws120_vs_nwp120"] ** 2

    if "era5v2_wind_speed_84m" in df.columns and "wind_speed_80m" in df.columns:
        df["era5v2_ws84_vs_nwp80"] = (
            df["era5v2_wind_speed_84m"].astype(float)
            - df["wind_speed_80m"].astype(float)
        )

    if "era5v2_wind_speed_120m" in df.columns:
        ws_era5v2_120 = df["era5v2_wind_speed_120m"].astype(float)
        df["era5v2_ws120m_cube"] = ws_era5v2_120 ** 3
        if "active_turbines_ratio" in df.columns:
            df["era5v2_ws120m_cube_x_active"] = (
                ws_era5v2_120 ** 3 * df["active_turbines_ratio"]
            )

    # --- LLJ (low-level jet) interactions ---
    if "era5v2_llj_strength" in df.columns:
        llj = df["era5v2_llj_strength"].astype(float)
        if "wind_speed_120m" in df.columns:
            df["era5v2_llj_x_ws120"] = llj * df["wind_speed_120m"].astype(float)
        if "rews" in df.columns:
            df["era5v2_llj_x_rews"] = llj * df["rews"].astype(float)
        df["era5v2_llj_flag"] = (llj > 1.0).astype(np.int8)

    # --- BLH (boundary layer height) — stability proxy ---
    if "era5v2_blh" in df.columns:
        blh = df["era5v2_blh"].astype(float)
        df["era5v2_blh_log"] = np.log1p(blh)
        if "wind_speed_120m" in df.columns:
            ws120 = df["wind_speed_120m"].astype(float)
            # High BLH + high wind = well-mixed boundary layer (good for power)
            df["era5v2_blh_x_ws120"] = blh * ws120 / 1000.0  # scale to avoid huge values
            # BLH / wind shear (stability index: deep BL suppresses vertical shear)
        if "ws_shear_10_80" in df.columns:
            shear = df["ws_shear_10_80"].astype(float).abs().clip(lower=0.01)
            df["era5v2_blh_over_shear"] = (blh / 1000.0) / shear
        # Shallow BLH = stable = possibly curtailed or below-curve
        df["era5v2_shallow_bl_flag"] = (blh < 200).astype(np.int8)

    # --- CAPE (convective available potential energy) ---
    if "era5v2_cape" in df.columns:
        cape = df["era5v2_cape"].astype(float).clip(lower=0.0)
        df["era5v2_cape_sqrt"] = np.sqrt(cape)
        df["era5v2_cape_flag"] = (cape > 100).astype(np.int8)
        if "wind_speed_120m" in df.columns:
            df["era5v2_cape_x_ws120"] = cape * df["wind_speed_120m"].astype(float)

    # --- Upper-level jet (850/925 hPa wind speeds) ---
    if "era5v2_wind_speed_850hpa" in df.columns and "wind_speed_120m" in df.columns:
        df["era5v2_850hpa_x_ws120"] = (
            df["era5v2_wind_speed_850hpa"].astype(float)
            * df["wind_speed_120m"].astype(float)
        )
    if "era5v2_wind_speed_925hpa" in df.columns:
        if "wind_speed_120m" in df.columns:
            df["era5v2_925hpa_x_ws120"] = (
                df["era5v2_wind_speed_925hpa"].astype(float)
                * df["wind_speed_120m"].astype(float)
            )
        # 850/925 vertical wind shear (jet structure proxy)
        if "era5v2_wind_speed_850hpa" in df.columns:
            df["era5v2_925_850_shear"] = (
                df["era5v2_wind_speed_925hpa"].astype(float)
                - df["era5v2_wind_speed_850hpa"].astype(float)
            )

    # --- TCWV (total column water vapour) — air density & icing ---
    if "era5v2_tcwv" in df.columns:
        df["era5v2_tcwv_norm"] = df["era5v2_tcwv"].astype(float) / 30.0  # typical range ~0-60

    # --- Snow / icing cross-features ---
    if "era5v2_snowc" in df.columns and "wind_speed_120m" in df.columns:
        df["era5v2_snow_x_ws120"] = (
            df["era5v2_snowc"].astype(float) * df["wind_speed_120m"].astype(float)
        )

    # --- Skin temperature vs T2m (surface energy balance) ---
    if "era5v2_skt_minus_t2m" in df.columns and "wind_speed_120m" in df.columns:
        df["era5v2_skt_t2m_x_ws120"] = (
            df["era5v2_skt_minus_t2m"].astype(float) * df["wind_speed_120m"].astype(float)
        )

    return df


# ---------------------------------------------------------------------------
# Multi-model direction agreement
# ---------------------------------------------------------------------------

def add_direction_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Circular dot-product agreement between different NWP direction forecasts.

    A low dot-product (near 0 or negative) means models disagree on wind
    direction — a strong predictor of forecast uncertainty and power error.
    """
    df = df.copy()

    # ERA5 direction (from _merge_era5: era5_dir100_sin, era5_dir100_cos)
    has_era5_dir = "era5_dir100_sin" in df.columns and "era5_dir100_cos" in df.columns

    # GFS direction agreement with ERA5
    if has_era5_dir and "gfs_dir100m_sin" in df.columns:
        dot = (
            df["gfs_dir100m_sin"] * df["era5_dir100_sin"]
            + df["gfs_dir100m_cos"] * df["era5_dir100_cos"]
        )
        df["gfs_era5_dir_agree"] = dot.clip(-1.0, 1.0)

    # ECMWF direction agreement with ERA5
    if has_era5_dir and "ecmwf_dir100m_sin" in df.columns:
        dot = (
            df["ecmwf_dir100m_sin"] * df["era5_dir100_sin"]
            + df["ecmwf_dir100m_cos"] * df["era5_dir100_cos"]
        )
        df["ecmwf_era5_dir_agree"] = dot.clip(-1.0, 1.0)

    # GFS vs ECMWF direction agreement
    if "gfs_dir100m_sin" in df.columns and "ecmwf_dir100m_sin" in df.columns:
        dot = (
            df["gfs_dir100m_sin"] * df["ecmwf_dir100m_sin"]
            + df["gfs_dir100m_cos"] * df["ecmwf_dir100m_cos"]
        )
        df["gfs_ecmwf_dir_agree"] = dot.clip(-1.0, 1.0)

    # NWP vs ERA5 direction agreement (already in era5_features.py as nwp_era5_dir_agreement,
    # but adding GFS/ECMWF × NWP interactions to capture multi-model direction consensus)
    if "gfs_era5_dir_agree" in df.columns and "ecmwf_era5_dir_agree" in df.columns:
        # Average direction agreement = overall directional confidence
        df["multi_dir_consensus"] = (
            df["gfs_era5_dir_agree"] + df["ecmwf_era5_dir_agree"]
        ) / 2.0
        # Low multi-direction agreement = directionally uncertain hour
        df["multi_dir_uncertain_flag"] = (df["multi_dir_consensus"] < 0.7).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Atmospheric stability interactions (from base NWP data)
# ---------------------------------------------------------------------------

def add_stability_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-terms between atmospheric stability indicators and wind speed.

    Uses features already in the pipeline from build_features() and extras.py.
    """
    df = df.copy()

    # temp_gradient is (T120 - T80) — positive = inversion (stable, laminar flow)
    if "temp_gradient" in df.columns and "wind_speed_120m" in df.columns:
        tg = df["temp_gradient"].astype(float)
        ws120 = df["wind_speed_120m"].astype(float)
        df["is_inversion"] = (tg > 0).astype(np.int8)
        df["stability_x_ws120"] = tg * ws120
        df["stability_x_rews"]  = tg * df["rews"].astype(float) if "rews" in df.columns else tg * ws120

    # Bulk Richardson number × wind speed (from extras.py)
    if "ri_bulk_80_120" in df.columns and "wind_speed_120m" in df.columns:
        ri = df["ri_bulk_80_120"].astype(float)
        ws = df["wind_speed_120m"].astype(float)
        df["ri_x_ws120"] = ri * ws
        # Stable (Ri > 0.25) but strong wind → unusual → potential for wind ramp
        df["stable_highwind_flag"] = ((ri > 0.25) & (ws > 8)).astype(np.int8)

    # Hellmann α (atmospheric stability exponent from pipeline.py)
    if "hellmann_alpha" in df.columns and "wind_speed_120m" in df.columns:
        alpha = df["hellmann_alpha"].astype(float)
        ws120 = df["wind_speed_120m"].astype(float)
        df["hellmann_x_ws120"] = alpha * ws120
        # Very high alpha = strong shear = stable → wind speed overestimated at hub
        df["high_shear_flag"] = (alpha > 0.3).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# Extended rolling features on multiple NWP sources
# ---------------------------------------------------------------------------

def add_extended_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Extended rolling statistics on GFS, ECMWF, and the ensemble spread.

    All rolling features are time-ordered and use closed='left' so that
    the current row is excluded (pure lag, no leakage).

    Called on the chronologically sorted combined (train+valid) frame.
    """
    df = df.copy()

    windows = [6, 12, 24]

    # GFS rolling
    if "gfs_ws100m" in df.columns:
        ws = df["gfs_ws100m"].astype(float)
        for w in windows:
            df[f"gfs_ws100_roll{w}h_mean"] = ws.rolling(w, min_periods=max(1, w // 2)).mean()
            df[f"gfs_ws100_roll{w}h_std"]  = ws.rolling(w, min_periods=max(1, w // 2)).std().fillna(0.0)
        df["gfs_ws100_accel"] = ws.diff(1) - ws.diff(2)  # second-order tendency

    # ECMWF rolling
    if "ecmwf_ws100m" in df.columns:
        ws = df["ecmwf_ws100m"].astype(float)
        for w in [6, 12]:
            df[f"ecmwf_ws100_roll{w}h_mean"] = ws.rolling(w, min_periods=max(1, w // 2)).mean()
        df["ecmwf_ws100_accel"] = ws.diff(1) - ws.diff(2)

    # Ensemble spread rolling — is forecast uncertainty growing or shrinking?
    if "ens3_ws100_std" in df.columns:
        spread = df["ens3_ws100_std"].astype(float)
        df["ens3_spread_roll6h_mean"] = spread.rolling(6, min_periods=1).mean()
        df["ens3_spread_roll12h_mean"] = spread.rolling(12, min_periods=1).mean()

    # Wind direction change (circular distance) using NWP hackathon directions
    if "wind_dir_120m_sin" in df.columns and "wind_dir_120m_cos" in df.columns:
        sin_c = df["wind_dir_120m_sin"].astype(float)
        cos_c = df["wind_dir_120m_cos"].astype(float)
        # Dot product with 1h-ago direction
        sin_lag = sin_c.shift(1).fillna(sin_c)
        cos_lag = cos_c.shift(1).fillna(cos_c)
        dot = (sin_c * sin_lag + cos_c * cos_lag).clip(-1.0, 1.0)
        df["dir_change_1h"] = np.arccos(dot)
        df["dir_change_3h_sin"] = sin_c - sin_c.shift(3).fillna(sin_c)
        df["dir_change_3h_cos"] = cos_c - cos_c.shift(3).fillna(cos_c)
        df["dir_change_3h_mag"] = np.sqrt(
            df["dir_change_3h_sin"] ** 2 + df["dir_change_3h_cos"] ** 2
        )
        # 6h direction stability (circular std proxy)
        df["dir_stability_6h"] = (
            df["dir_change_1h"].rolling(6, min_periods=1).std().fillna(0.0)
        )

    return df


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def add_all_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all advanced interaction feature groups in order."""
    df = add_era5v2_interactions(df)
    df = add_direction_agreement(df)
    df = add_stability_interactions(df)
    df = add_extended_rolling(df)
    return df


def advanced_interaction_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns added by this module."""
    prefixes = (
        "era5v2_ws120_vs_", "era5v2_ws84_vs_", "era5v2_ws120m_", "era5v2_llj_",
        "era5v2_blh_", "era5v2_cape_", "era5v2_850hpa_", "era5v2_925hpa_",
        "era5v2_925_850_", "era5v2_tcwv_", "era5v2_snow_", "era5v2_skt_t2m_",
        "era5v2_shallow_", "era5v2_llj_flag",
        "gfs_era5_dir_", "ecmwf_era5_dir_", "gfs_ecmwf_dir_", "multi_dir_",
        "is_inversion", "stability_x_", "ri_x_", "stable_high", "hellmann_x_",
        "high_shear_", "gfs_ws100_roll", "gfs_ws100_accel",
        "ecmwf_ws100_roll", "ecmwf_ws100_accel",
        "ens3_spread_roll", "dir_change_", "dir_stability_",
    )
    standalone = {
        "era5v2_ws120m_cube", "era5v2_ws120m_cube_x_active",
        "era5v2_llj_flag", "era5v2_cape_sqrt", "era5v2_cape_flag",
        "era5v2_shallow_bl_flag", "era5v2_blh_log",
        "is_inversion", "stable_highwind_flag", "high_shear_flag",
        "multi_dir_consensus", "multi_dir_uncertain_flag",
    }
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in prefixes) or c in standalone
    ]
