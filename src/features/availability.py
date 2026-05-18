"""Walk-forward seasonal availability features.

Motivation
----------
The Fold-5 diagnostic (scripts/diagnose_v32_fold5.py) showed:

  - 56 % of all blend error in the 7-12 m/s band
  - +1.32 MW positive bias in the 12-17 m/s "rated" band
  - The 20 worst hours all had wind ~10 m/s but actual output 5-25 MW —
    classic curtailment / unavailability events not captured by the slow-
    moving ``n_repair`` column.

Earlier curtailment attempts (v8/v9 in ``src/training/_archive``) used
*current-row* curtailment flags as features and hurt Fold-5. The likely
reason: those attempts were target-leaky in cross-validation (the flag
was computed using the row's own actual production).

This module fixes the methodology:

  1. Compute a per-row "underproduction" residual on the FULL training
     set (target_mw - p_expected_isotonic), where ``p_expected_isotonic``
     is fit per-fold on rows older than ``train_end``.
  2. Aggregate the residual by ``(month, hour_of_day)`` using *only* rows
     STRICTLY BEFORE the current row's timestamp (walk-forward expanding
     window).
  3. Expose three features per row:
       - ``avail_underprod_rate_mh``: average past underproduction at this
         (month, hour) cell, in MW. Negative = past availability
         systematically below physics.
       - ``avail_underprod_count_mh``: how many past rows informed it
         (information-content gauge for the booster).
       - ``avail_underprod_rate_30d``: 30-day trailing mean of underprod
         (catches week-scale curtailment regimes).

For the validation/test set (Q1 2026), the walk-forward stops at the end
of the training set so all rows in Q1 2026 see the *same* (month, hour)
table — that's the appropriate behaviour because at forecast time we
don't yet know Q1 2026 actuals.

Intra-fold leak fix (``freeze_after_ts``)
-----------------------------------------
Without the ``freeze_after_ts`` parameter, a Fold-5 validation row at
(Jan, hr=10, 2025-01-15) would accumulate (Jan, hr=10) observations from
2025-01-01..14 — other rows in the same validation split — inflating the
OOF Fold-5 nMAE vs the leaderboard.  Pass ``fold.train_end`` as
``freeze_after_ts`` to restrict underprod observations to
rows <= fold.train_end.  Training scripts (v34+) should do this for every
cross-validation fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL


def fit_p_expected_isotonic(
    df: pd.DataFrame,
    *,
    wind_col: str = "wind_speed_120m",
    impossible_col: str = "_is_impossible",
) -> IsotonicRegression:
    """Fit an isotonic ``ws -> mw`` baseline on clean training rows only."""
    mask = df[TARGET_COL].notna() & (df[TARGET_COL] >= 0)
    if impossible_col in df.columns:
        mask &= ~df[impossible_col].astype(bool)
    valid = df.loc[mask, [wind_col, TARGET_COL]].dropna()
    ir = IsotonicRegression(
        y_min=0.0, y_max=float(CAPACITY_MW),
        out_of_bounds="clip", increasing=True,
    )
    ir.fit(valid[wind_col].to_numpy(), valid[TARGET_COL].to_numpy())
    return ir


def add_walk_forward_availability(
    df: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    wind_col: str = "wind_speed_120m",
    impossible_col: str = "_is_impossible",
    window_days: int = 30,
    freeze_after_ts: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """Add walk-forward availability features.

    Parameters
    ----------
    df:
        Combined train + valid frame, sorted ascending by ``TIMESTAMP_COL``.
        Must contain ``TARGET_COL`` (NaN fine for valid rows), ``wind_col``,
        and optionally ``_is_impossible``.
    train_end:
        Used to fit the isotonic baseline (only rows <= train_end used).
    freeze_after_ts:
        If provided, rows AFTER this timestamp do not contribute underprod
        observations to the (month, hour) cumsum or the 30-day rolling
        window.  Pass ``fold.train_end`` for honest cross-validation to
        prevent intra-fold leakage (validation rows of one fold polluting
        the (month, hour) stats for later validation rows in the same fold).
    wind_col, impossible_col, window_days:
        See module docstring.

    Returns a copy of ``df`` with five new columns:
        avail_p_expected_mw, avail_underprod_mw,
        avail_underprod_rate_mh, avail_underprod_count_mh,
        avail_underprod_rate_30d
    """
    df = df.copy()
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # ── 1. Isotonic baseline (fit on training rows only) ─────────────
    train_mask = df[TIMESTAMP_COL] <= train_end
    ir = fit_p_expected_isotonic(
        df.loc[train_mask],
        wind_col=wind_col,
        impossible_col=impossible_col,
    )

    # ── 2. Apply baseline everywhere ─────────────────────────────────
    df["avail_p_expected_mw"] = ir.predict(df[wind_col].to_numpy())
    df["avail_p_expected_mw"] = df["avail_p_expected_mw"].clip(0.0, CAPACITY_MW)

    # ── 3. Underproduction residual ───────────────────────────────────
    # Only set for rows within the "observable" window.
    # freeze_after_ts gates out val rows so they never write to the table.
    df["avail_underprod_mw"] = np.nan
    observable_mask = train_mask.copy()
    if freeze_after_ts is not None:
        observable_mask &= df[TIMESTAMP_COL] <= freeze_after_ts
    valid_obs = observable_mask & df[TARGET_COL].notna()
    if impossible_col in df.columns:
        valid_obs &= ~df[impossible_col].astype(bool)
    df.loc[valid_obs, "avail_underprod_mw"] = (
        df.loc[valid_obs, TARGET_COL]
        - df.loc[valid_obs, "avail_p_expected_mw"]
    )

    # ── 4. Walk-forward (month, hour) cumulative mean ─────────────────
    df["_month"] = df[TIMESTAMP_COL].dt.month
    df["_hour"]  = df[TIMESTAMP_COL].dt.hour
    df["_underprod_for_cum"] = df["avail_underprod_mw"].fillna(0.0)
    df["_count_for_cum"]     = df["avail_underprod_mw"].notna().astype(int)

    df["_cum_sum"]   = (
        df.groupby(["_month", "_hour"], sort=False)["_underprod_for_cum"].cumsum()
    )
    df["_cum_count"] = (
        df.groupby(["_month", "_hour"], sort=False)["_count_for_cum"].cumsum()
    )
    # Shift WITHIN each (month, hour) group so each row sees strictly
    # past observations only.
    df["_cum_sum"] = (
        df.groupby(["_month", "_hour"], sort=False)["_cum_sum"]
        .shift(1).fillna(0.0)
    )
    df["_cum_count"] = (
        df.groupby(["_month", "_hour"], sort=False)["_cum_count"]
        .shift(1).fillna(0).astype(int)
    )

    df["avail_underprod_rate_mh"] = np.where(
        df["_cum_count"] > 0,
        df["_cum_sum"] / df["_cum_count"],
        0.0,
    )
    df["avail_underprod_count_mh"] = df["_cum_count"]

    # ── 5. 30-day trailing mean ───────────────────────────────────────
    s      = df.set_index(TIMESTAMP_COL)["avail_underprod_mw"]
    rolled = s.rolling(f"{window_days}D", closed="left").mean()
    df["avail_underprod_rate_30d"] = rolled.fillna(0.0).to_numpy()

    # ── Cleanup ───────────────────────────────────────────────────────
    df = df.drop(columns=[
        "_month", "_hour", "_underprod_for_cum", "_count_for_cum",
        "_cum_sum", "_cum_count",
    ])
    return df


def availability_columns() -> list[str]:
    """Return the names of the columns added by ``add_walk_forward_availability``."""
    return [
        "avail_p_expected_mw",
        "avail_underprod_mw",
        "avail_underprod_rate_mh",
        "avail_underprod_count_mh",
        "avail_underprod_rate_30d",
    ]
