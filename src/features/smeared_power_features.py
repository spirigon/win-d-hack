"""Smeared (uncertainty-aware) theoretical power features.

Motivation
----------
NWP wind-speed forecasts carry ~1-4 m/s uncertainty.  The standard theoretical
power P(v_forecast) ignores this: because the power curve is convex in the
steep zone (7-12 m/s), Jensen's inequality gives

    E[P(v)] > P(E[v])

so P(v_forecast) *underestimates* expected power when the true speed could be
v ± delta_V.  This module computes:

    smear_P(v, delta) = (P(v - delta) + P(v + delta)) / 2

for several delta values, and the excess over standard theoretical power:

    smear_excess(v, delta) = smear_P(v, delta) - P(v)

smear_excess > 0 in the cubic regime (7-12 m/s) and ~0 at rated power — it's
a trainable proxy for the Jensen correction the model needs to apply.

Delta values (m/s) — adjust as needed:
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import CAPACITY_MW
from src.features.datasheet_power_curve import per_turbine_power_kw

# ── Configuration ─────────────────────────────────────────────────────────────
DELTA_V_LIST: list[float] = [1.0, 2.0, 3.0, 4.0]   # m/s; edit freely
# ──────────────────────────────────────────────────────────────────────────────

_WS_COL   = "wind_speed_120m"
_DENS_COL = "air_density"


def add_smeared_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add smeared theoretical power features for each delta in DELTA_V_LIST.

    For each delta_V:
      smear_p{d}_per_turb_kw — mean of P(v-δ) and P(v+δ) per turbine (kW)
      smear_p{d}_farm_mw     — farm-level smeared power (MW)
      smear_p{d}_excess_mw   — smear_farm - standard P(v) farm (MW); > 0 in cubic zone
    """
    if _WS_COL not in df.columns or _DENS_COL not in df.columns:
        return df

    df = df.copy()
    ws      = df[_WS_COL].to_numpy(dtype=np.float64)
    density = df[_DENS_COL].to_numpy(dtype=np.float64)
    active  = df["active_turbines"].to_numpy(dtype=np.float64)

    p_std_kw = per_turbine_power_kw(ws, density)
    p_std_mw = np.clip(p_std_kw * active / 1000.0, 0.0, CAPACITY_MW)

    for delta in DELTA_V_LIST:
        tag = str(delta).replace(".", "")   # "1", "2", "3", "4"
        p_lo_kw = per_turbine_power_kw(ws - delta, density)
        p_hi_kw = per_turbine_power_kw(ws + delta, density)
        smear_kw = (p_lo_kw + p_hi_kw) / 2.0
        smear_mw = np.clip(smear_kw * active / 1000.0, 0.0, CAPACITY_MW)

        df[f"smear_p{tag}_per_turb_kw"] = smear_kw.astype(np.float32)
        df[f"smear_p{tag}_farm_mw"]     = smear_mw.astype(np.float32)
        df[f"smear_p{tag}_excess_mw"]   = (smear_mw - p_std_mw).astype(np.float32)

    return df


def smeared_power_columns(df: pd.DataFrame) -> list[str]:
    """Return smeared-power feature columns present in df."""
    prefixes = tuple(
        f"smear_p{str(d).replace('.', '')}_" for d in DELTA_V_LIST
    )
    return [c for c in df.columns if c.startswith(prefixes)]
