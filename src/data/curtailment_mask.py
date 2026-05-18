"""Identify curtailment / unavailability events that the legacy outlier
rules miss.

The Fold-5 diagnostic on v32 (``scripts/diagnose_v32_fold5.py``) showed
that 20+ hours in the Fold-5 validation window are catastrophic over-
predictions: NWP says 9-12 m/s wind, the model expects 50-70 MW, but
actual production was 5-25 MW. Inspecting these rows shows persistent
multi-hour underproduction (March 11-12, 2025) at full hub wind. They
look like operator-driven curtailment or grid-side restrictions —
events the model cannot learn to predict from weather alone.

The legacy ``rated_plateau_shortfall`` rule (``ws > 14`` AND
``active >= 24`` AND ``target < 60``) is too narrow: it requires both
storm-band wind AND near-full availability. These curtailment events
occur in mid-band wind and at any availability.

This module exposes a separate, **opt-in** mask so the legacy
reproducibility chain (used by v32 / v34 / v34.1 / etc.) is not
disturbed. Train scripts that want to test the cleaner mask call
:func:`identify_curtailment_rows` and OR it into ``_is_impossible``
before computing sample weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.schema import (
    CAPACITY_MW,
    TARGET_COL,
    TURBINES_IN_MAINTENANCE_COL,
)


@dataclass(frozen=True)
class CurtailmentConfig:
    """Threshold parameters for the curtailment detector.

    Defaults are chosen to be **conservative** — we'd rather miss a
    curtailment row than mis-flag a real production hour.

    Attributes
    ----------
    min_wind_ms:
        Below this hub wind we never flag — no expected production at
        cut-in.
    fraction_threshold:
        Flag a row if ``actual / expected_clean < fraction_threshold``.
        Default 0.4 means actual is less than 40% of weather-implied
        expectation.
    min_expected_mw:
        Don't flag low-magnitude rows where small absolute errors look
        large in ratio. Defaults to 15 MW (≈17% of capacity).
    min_run_length_hours:
        Curtailment events are persistent (multi-hour). Singleton flags
        are usually the model's own variance, not curtailment. We
        therefore require a run of length >= this to be confirmed as
        curtailment.
    expected_curve_eval:
        How to compute the "expected" baseline. ``"isotonic"`` fits an
        isotonic on (ws, target) on rows that pass *other* impossibility
        filters; ``"datasheet"`` uses the manufacturer table.
    """
    min_wind_ms: float = 5.0
    fraction_threshold: float = 0.4
    min_expected_mw: float = 15.0
    min_run_length_hours: int = 3
    expected_curve_eval: str = "isotonic"


def _isotonic_expected(
    ws: np.ndarray, target: np.ndarray, *, fit_mask: np.ndarray
) -> np.ndarray:
    """Predict expected MW from wind speed via isotonic regression.

    The isotonic is fitted on rows where ``fit_mask=True`` (training-only
    clean rows) so the resulting baseline doesn't try to interpolate
    through the very curtailment events we want to flag.
    """
    ir = IsotonicRegression(
        y_min=0.0, y_max=float(CAPACITY_MW),
        out_of_bounds="clip",
        increasing=True,
    )
    ir.fit(ws[fit_mask], target[fit_mask])
    return ir.predict(ws)


def _datasheet_expected(ws: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Use manufacturer datasheet PC × active turbines."""
    from src.features.datasheet_power_curve import per_turbine_power_kw
    n_repair = df[TURBINES_IN_MAINTENANCE_COL].fillna(0).to_numpy()
    n_active = (26 - n_repair).clip(min=0)
    rho = df.get("air_density", pd.Series(1.225, index=df.index)).fillna(1.225).to_numpy()
    p_per = per_turbine_power_kw(ws, rho)
    return np.clip(p_per * n_active / 1000.0, 0.0, CAPACITY_MW)


def identify_curtailment_rows(
    df: pd.DataFrame,
    *,
    cfg: CurtailmentConfig | None = None,
    pre_existing_impossible: pd.Series | None = None,
    wind_col: str = "wind_speed_120m",
) -> tuple[pd.Series, pd.DataFrame]:
    """Return a boolean mask of curtailment rows + diagnostic dataframe.

    Parameters
    ----------
    df:
        Training frame. Must contain ``TARGET_COL``, ``wind_col``,
        ``TURBINES_IN_MAINTENANCE_COL``. Optionally ``air_density`` for
        the datasheet baseline.
    cfg:
        Threshold configuration. ``None`` → default
        :class:`CurtailmentConfig`.
    pre_existing_impossible:
        Boolean series (aligned with ``df.index``) marking rows already
        flagged by other rules. Excluded from the isotonic fit so the
        baseline is built only from plausible rows.
    wind_col:
        Hub wind column name.

    Returns
    -------
    (mask, diag_df):
        ``mask`` — pandas Series of bools indexed like ``df``.
        ``diag_df`` — same length, columns ``ws``, ``target``,
            ``expected``, ``ratio``, ``flagged_pre_run``, ``flagged``.
    """
    cfg = cfg or CurtailmentConfig()
    df = df.reset_index(drop=True)
    if pre_existing_impossible is None:
        pre_existing_impossible = pd.Series(False, index=df.index)
    else:
        pre_existing_impossible = pre_existing_impossible.reset_index(drop=True)

    ws = df[wind_col].fillna(0.0).to_numpy(dtype=float)
    target = df[TARGET_COL].fillna(0.0).to_numpy(dtype=float)

    # Build expected curve.
    if cfg.expected_curve_eval == "isotonic":
        fit_mask = (~pre_existing_impossible.to_numpy()) & (df[TARGET_COL].notna()) & (target >= 0)
        expected = _isotonic_expected(ws, target, fit_mask=fit_mask)
    elif cfg.expected_curve_eval == "datasheet":
        expected = _datasheet_expected(ws, df)
    else:
        raise ValueError(f"Unknown expected_curve_eval={cfg.expected_curve_eval!r}")

    # Pre-flag: per-row underproduction.
    safe_expected = np.where(expected > cfg.min_expected_mw, expected, np.nan)
    ratio = target / safe_expected
    pre_flag = (
        (ws >= cfg.min_wind_ms)
        & (expected >= cfg.min_expected_mw)
        & (ratio < cfg.fraction_threshold)
    )
    pre_flag = pd.Series(pre_flag, index=df.index).fillna(False)

    # Run-length filter: keep only pre-flag rows that are part of a run
    # of length >= min_run_length_hours.
    flagged = pre_flag.copy()
    if cfg.min_run_length_hours > 1:
        # Compute consecutive-True run lengths.
        run_id = (pre_flag != pre_flag.shift()).cumsum()
        run_size = pre_flag.groupby(run_id).transform("sum")  # only counts True; runs of False sum to 0
        # A run is "kept" if pre_flag is True AND its run size meets the threshold.
        flagged = pre_flag & (run_size >= cfg.min_run_length_hours)

    diag = pd.DataFrame({
        "ws": ws,
        "target": target,
        "expected": expected,
        "ratio": ratio,
        "flagged_pre_run": pre_flag.to_numpy(),
        "flagged": flagged.to_numpy(),
    })
    return flagged, diag


def summarize_curtailment(diag: pd.DataFrame, ts: pd.Series | None = None) -> dict:
    """One-line stats about the curtailment mask, useful for the script header."""
    n = len(diag)
    n_pre = int(diag["flagged_pre_run"].sum())
    n_flagged = int(diag["flagged"].sum())
    summary = {
        "total_rows": int(n),
        "pre_run_flagged": n_pre,
        "post_run_flagged": n_flagged,
        "pct_flagged": (n_flagged / n * 100.0) if n else 0.0,
        "mean_target_on_flagged": float(diag.loc[diag["flagged"], "target"].mean()) if n_flagged else 0.0,
        "mean_expected_on_flagged": float(diag.loc[diag["flagged"], "expected"].mean()) if n_flagged else 0.0,
    }
    return summary
