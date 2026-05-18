"""Advanced turbulence, ensemble rank, and density-corrected features.

The ws 7-10 m/s zone has MAE=10.29 MW with 35% over-prediction and 34%
under-prediction — zero mean bias but high variance. The root cause is
that at these wind speeds, small turbulence changes and air density
differences cause large power swings the models can't see from mean
wind speed alone.

This module adds:
    Multi-model turbulence intensity
        gfs/ecmwf gust ratios, cross-model TI disagreement
    NWP ensemble rank
        where hackathon NWP sits among all available models (high rank →
        likely over-estimated → model should correct down)
    Density-corrected WPD from GFS and ECMWF
        using each model's own T + P + ws to compute accurate wind power
        density (captures density variability independent of ERA5)
    Steep-zone interaction features
        spread × ws_steep_zone flag — uncertainty matters most there
    Era5v2 v_eff at exact hub height
        era5v2 air density × era5v2 ws120m (most physics-consistent)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.physics import compute_air_density, compute_v_eff, compute_wpd

TIMESTAMP_COL = "METEOFORECASTHOUR_OPENM_Datetime"
_CUT_IN  = 3.0   # m/s
_STEEP_LO = 7.0  # m/s — start of steep PC slope
_STEEP_HI = 10.0 # m/s — end of steep PC slope
_RATED    = 12.0 # m/s — near rated wind speed


# ---------------------------------------------------------------------------
# Multi-model turbulence intensity
# ---------------------------------------------------------------------------

def add_turbulence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gust-based turbulence intensity from all available NWP sources."""
    df = df.copy()

    # Per-model TI proxy: gust / wind_speed
    ti_series = {}

    # Hackathon NWP
    if "wind_speed_10m" in df.columns and "wind_gust_10m" in df.columns:
        nwp_ws = df["wind_speed_10m"].astype(float).clip(lower=0.5)
        nwp_gust = df["wind_gust_10m"].astype(float)
        df["nwp_ti_10m"] = (nwp_gust / nwp_ws).clip(1.0, 5.0)
        ti_series["nwp"] = df["nwp_ti_10m"]
    elif "gust_ratio_10m" in df.columns:
        ti_series["nwp"] = df["gust_ratio_10m"].astype(float)

    # GFS
    if "gfs_gust10m" in df.columns and "gfs_ws10m" in df.columns:
        gfs_ws = df["gfs_ws10m"].astype(float).clip(lower=0.5)
        gfs_gust = df["gfs_gust10m"].astype(float)
        df["gfs_ti_10m"] = (gfs_gust / gfs_ws).clip(1.0, 5.0)
        ti_series["gfs"] = df["gfs_ti_10m"]

    # ECMWF
    if "ecmwf_gust10m" in df.columns and "ecmwf_ws10m" in df.columns:
        ecmwf_ws = df["ecmwf_ws10m"].astype(float).clip(lower=0.5)
        ecmwf_gust = df["ecmwf_gust10m"].astype(float)
        df["ecmwf_ti_10m"] = (ecmwf_gust / ecmwf_ws).clip(1.0, 5.0)
        ti_series["ecmwf"] = df["ecmwf_ti_10m"]

    # ERA5
    if "era5_wind_gusts_10m" in df.columns and "era5_wind_speed_10m" in df.columns:
        era5_ws = df["era5_wind_speed_10m"].astype(float).clip(lower=0.5)
        era5_gust = df["era5_wind_gusts_10m"].astype(float)
        df["era5_ti_10m"] = (era5_gust / era5_ws).clip(1.0, 5.0)
        ti_series["era5"] = df["era5_ti_10m"]

    # ICON-EU
    if "icon_gusts10m" in df.columns and "icon_ws10m" in df.columns:
        icon_ws = df["icon_ws10m"].astype(float).clip(lower=0.5)
        icon_gust = df["icon_gusts10m"].astype(float)
        df["icon_ti_10m"] = (icon_gust / icon_ws).clip(1.0, 5.0)
        ti_series["icon"] = df["icon_ti_10m"]

    # Multi-model TI ensemble
    if len(ti_series) >= 2:
        ti_mat = pd.concat(ti_series.values(), axis=1).to_numpy(dtype=float)
        df["ti_multi_mean"] = np.nanmean(ti_mat, axis=1)
        df["ti_multi_std"]  = np.nanstd(ti_mat,  axis=1)
        df["ti_multi_max"]  = np.nanmax(ti_mat,  axis=1)

        # High TI = turbulent flow = more scatter around mean power curve
        df["high_turbulence_flag"] = (df["ti_multi_mean"] > 1.8).astype(np.int8)

        # TI × steep zone interaction (turbulence matters MOST in 7-10 zone)
        if "wind_speed_120m" in df.columns:
            steep = ((df["wind_speed_120m"] >= _STEEP_LO) &
                     (df["wind_speed_120m"] < _STEEP_HI)).astype(float)
            df["ti_x_steep_zone"] = df["ti_multi_mean"] * steep
            df["ti_spread_x_steep"] = df["ti_multi_std"] * steep

    # Gust excess at hub height proxy
    if "gust_excess_10m" in df.columns and "wind_speed_120m" in df.columns:
        df["gust_excess_x_ws120"] = (
            df["gust_excess_10m"].astype(float) * df["wind_speed_120m"].astype(float)
        )

    return df


# ---------------------------------------------------------------------------
# NWP ensemble rank — where does hackathon NWP fall?
# ---------------------------------------------------------------------------

def add_ensemble_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank hackathon NWP 120m wind among all available hub-height estimates.

    If the hackathon NWP is consistently the highest (or lowest) among all
    models, that's a strong bias signal: extreme rank → likely mis-estimated.
    """
    df = df.copy()

    if "wind_speed_120m" not in df.columns:
        return df

    # Project hackathon 120m → 100m for comparison with other models
    nwp_at_100 = df["wind_speed_120m"].astype(float) * (100.0 / 120.0) ** 0.14

    sources: dict[str, pd.Series] = {"nwp": nwp_at_100}
    if "era5_wind_speed_100m" in df.columns:
        sources["era5"]  = df["era5_wind_speed_100m"].astype(float)
    if "gfs_ws100m" in df.columns:
        sources["gfs"]   = df["gfs_ws100m"].astype(float)
    if "ecmwf_ws100m" in df.columns:
        sources["ecmwf"] = df["ecmwf_ws100m"].astype(float)
    if "icon_ws100m" in df.columns:
        sources["icon"]  = df["icon_ws100m"].astype(float)
    if "nasa_ws50m" in df.columns:
        # MERRA-2 50m projected to 100m using 1/7 law
        sources["nasa"]  = df["nasa_ws50m"].astype(float) * (100.0 / 50.0) ** (1.0/7.0)

    if len(sources) < 3:
        return df

    mat = pd.concat(sources.values(), axis=1).to_numpy(dtype=float)
    n_models = mat.shape[1]
    nwp_idx = list(sources.keys()).index("nwp")

    # Rank of NWP in each row (0 = lowest, n_models-1 = highest)
    ranks = np.apply_along_axis(
        lambda row: np.argsort(np.argsort(np.where(np.isnan(row), np.nanmedian(row), row))),
        axis=1, arr=mat,
    )
    nwp_rank = ranks[:, nwp_idx].astype(float)

    # Normalize to [0, 1]
    df["nwp_rank_in_ens"] = nwp_rank / (n_models - 1)
    # Centered: positive = NWP above median model, negative = below
    df["nwp_rank_centered"] = df["nwp_rank_in_ens"] - 0.5
    # High rank = likely over-estimated; flag extreme ranks
    df["nwp_is_highest_flag"]  = (nwp_rank == n_models - 1).astype(np.int8)
    df["nwp_is_lowest_flag"]   = (nwp_rank == 0).astype(np.int8)

    # NWP deviation from ensemble median (direction + magnitude)
    ens_median = np.nanmedian(mat, axis=1)
    df["nwp_vs_ens_median"]    = nwp_at_100.to_numpy() - ens_median
    df["nwp_vs_ens_median_sq"] = df["nwp_vs_ens_median"] ** 2  # non-linear signal

    # Steep-zone amplification: being the outlier matters most in 7-10 zone
    if "wind_speed_120m" in df.columns:
        steep = ((df["wind_speed_120m"] >= _STEEP_LO) &
                 (df["wind_speed_120m"] < _STEEP_HI)).astype(float)
        df["nwp_rank_x_steep"]    = df["nwp_rank_in_ens"] * steep
        df["nwp_outlier_x_steep"] = df["nwp_vs_ens_median"].abs() * steep

    return df


# ---------------------------------------------------------------------------
# Density-corrected WPD from GFS and ECMWF
# ---------------------------------------------------------------------------

def add_model_wpd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute wind power density for GFS and ECMWF using their own T+P+ws.

    Each NWP model has a different temperature and pressure estimate. The
    resulting air density difference can be 1-3% — enough to shift power
    estimates by 1-2 MW in high-wind conditions. This captures physical
    differences between models beyond just wind speed.
    """
    df = df.copy()

    # GFS WPD (pressure in Pa: gfs_msl is in hPa, convert)
    if all(c in df.columns for c in ["gfs_ws100m", "gfs_t2m", "gfs_msl"]):
        # gfs_msl is in hPa — compute_air_density expects hPa
        gfs_rho = compute_air_density(df["gfs_msl"].astype(float),
                                      df["gfs_t2m"].astype(float))
        gfs_ws = df["gfs_ws100m"].astype(float)
        df["gfs_air_density"] = gfs_rho
        df["gfs_v_eff"]       = compute_v_eff(gfs_ws, gfs_rho)
        df["gfs_wpd"]         = compute_wpd(gfs_ws, gfs_rho)
        df["gfs_v_eff_cube"]  = df["gfs_v_eff"] ** 3

        # GFS density-adjusted bias vs ERA5 WS
        if "era5_wind_speed_100m" in df.columns and "air_density" in df.columns:
            df["gfs_wpd_vs_era5_wpd"] = df["gfs_wpd"] - compute_wpd(
                df["era5_wind_speed_100m"].astype(float), df["air_density"].astype(float)
            )

    # ECMWF WPD
    if all(c in df.columns for c in ["ecmwf_ws100m", "ecmwf_t2m", "ecmwf_msl"]):
        ecmwf_rho = compute_air_density(df["ecmwf_msl"].astype(float),
                                        df["ecmwf_t2m"].astype(float))
        ecmwf_ws = df["ecmwf_ws100m"].astype(float)
        df["ecmwf_air_density"] = ecmwf_rho
        df["ecmwf_v_eff"]       = compute_v_eff(ecmwf_ws, ecmwf_rho)
        df["ecmwf_wpd"]         = compute_wpd(ecmwf_ws, ecmwf_rho)
        df["ecmwf_v_eff_cube"]  = df["ecmwf_v_eff"] ** 3

    # Cross-model WPD consensus (most physically grounded ensemble signal)
    wpd_sources = [c for c in ["wpd_120m", "gfs_wpd", "ecmwf_wpd", "era5_wpd"] if c in df.columns]
    if len(wpd_sources) >= 2:
        wpd_mat = df[wpd_sources].astype(float).to_numpy()
        df["wpd_ens_mean"]  = np.nanmean(wpd_mat, axis=1)
        df["wpd_ens_std"]   = np.nanstd(wpd_mat,  axis=1)
        df["wpd_ens_range"] = np.nanmax(wpd_mat,  axis=1) - np.nanmin(wpd_mat, axis=1)

    # ERA5v2 v_eff at exact hub height — most physics-consistent single-source estimate
    if "era5v2_air_density" in df.columns and "era5v2_wind_speed_120m" in df.columns:
        era5v2_rho = df["era5v2_air_density"].astype(float)
        era5v2_ws  = df["era5v2_wind_speed_120m"].astype(float)
        df["era5v2_v_eff_120m"]      = compute_v_eff(era5v2_ws, era5v2_rho)
        df["era5v2_v_eff_120m_cube"] = df["era5v2_v_eff_120m"] ** 3
        df["era5v2_wpd_120m"]        = compute_wpd(era5v2_ws, era5v2_rho)
        if "active_turbines_ratio" in df.columns:
            df["era5v2_v_eff_120m_cube_x_active"] = (
                df["era5v2_v_eff_120m"] ** 3 * df["active_turbines_ratio"]
            )

    return df


# ---------------------------------------------------------------------------
# Wind regime and steep-zone features
# ---------------------------------------------------------------------------

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wind regime indicators and steep-zone interaction features.

    The ws 7-10 zone has zero mean bias but 10.3 MW MAE — high variance,
    not systematic error. Better regime detection helps the model know
    when it should predict conservatively vs aggressively.
    """
    df = df.copy()

    if "wind_speed_120m" not in df.columns:
        return df

    ws = df["wind_speed_120m"].astype(float)

    # Soft regime membership (piecewise linear, not binary)
    df["regime_sub_cutin"]  = (1.0 - (ws / _CUT_IN).clip(0, 1))         # 1 below cut-in
    df["regime_steep"]      = (                                            # 1 in 7-10 zone
        ((ws - _STEEP_LO) / (_STEEP_HI - _STEEP_LO)).clip(0, 1) *
        (1.0 - ((ws - _STEEP_HI) / 2.0).clip(0, 1))
    )
    df["regime_rated"]      = ((ws - _RATED) / 2.0).clip(0, 1)           # 1 above rated
    df["regime_steep_flag"] = ((ws >= _STEEP_LO) & (ws < _STEEP_HI)).astype(np.int8)
    df["regime_rated_flag"] = (ws >= _RATED).astype(np.int8)

    # Steep zone × ensemble spread (uncertainty is most damaging here)
    if "ens3_ws100_std" in df.columns:
        df["spread_x_steep"] = df["ens3_ws100_std"] * df["regime_steep"]
        df["spread_x_rated"] = df["ens3_ws100_std"] * df["regime_rated"]

    # Steep zone × stability (unstable boundary layer → more variance in steep zone)
    if "hellmann_alpha" in df.columns:
        df["alpha_x_steep"] = df["hellmann_alpha"].astype(float) * df["regime_steep"]

    if "ri_bulk_80_120" in df.columns:
        df["ri_x_steep"] = df["ri_bulk_80_120"].astype(float) * df["regime_steep"]

    # Night-hour density boost (nocturnal boundary layer → higher air density → more power)
    # Night hours (20-23, 0-3) have highest MAE — interaction helps
    if TIMESTAMP_COL in df.columns:
        hour = df[TIMESTAMP_COL].dt.hour
        df["is_night"] = ((hour >= 20) | (hour <= 3)).astype(np.int8)
        if "air_density" in df.columns:
            df["night_density"] = df["is_night"] * df["air_density"].astype(float)
        if "wind_speed_120m" in df.columns:
            df["night_x_ws120"] = df["is_night"] * ws

    return df


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def add_all_turbulence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all turbulence / advanced feature groups."""
    df = add_turbulence_features(df)
    df = add_ensemble_rank_features(df)
    df = add_model_wpd_features(df)
    df = add_regime_features(df)
    return df


def turbulence_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns added by this module."""
    prefixes = (
        "nwp_ti_", "gfs_ti_", "ecmwf_ti_", "era5_ti_", "icon_ti_",
        "ti_multi_", "high_turbulence_", "ti_x_", "ti_spread_",
        "gust_excess_x_",
        "nwp_rank_", "nwp_is_", "nwp_vs_ens_", "nwp_outlier_",
        "gfs_air_", "gfs_v_eff", "gfs_wpd", "ecmwf_air_", "ecmwf_v_eff",
        "ecmwf_wpd", "wpd_ens_", "era5v2_v_eff_120m", "era5v2_wpd_120m",
        "regime_", "spread_x_", "alpha_x_", "ri_x_", "is_night",
        "night_density", "night_x_",
    )
    standalone = {
        "nwp_rank_centered", "nwp_rank_x_steep", "nwp_vs_ens_median_sq",
        "gfs_wpd_vs_era5_wpd", "era5v2_v_eff_120m_cube_x_active",
    }
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in prefixes) or c in standalone
    ]
