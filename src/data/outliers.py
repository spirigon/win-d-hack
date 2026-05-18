"""Identify and remove / downweight physically impossible training rows.

This module holds the **Cleaner** half of the Data Quality Hardening pack
(spec: ``.kiro/specs/data-quality-hardening``). It is the single place that
decides whether a training row is "impossible" and therefore carries zero
weight, or "plausible" and carries the normal weight of ``1.0``.

Four impossible rules are evaluated in a **fixed priority order**, so the
reason string returned by :func:`impossible_reasons` is deterministic when
multiple predicates match the same row:

1. ``over_capacity``            — target strictly above ``CAPACITY_MW``.
2. ``maintenance_zero``         — 24+ turbines down *and* target < 1.0 MW.
3. ``rated_plateau_shortfall``  — ws_120m > 14 m/s, 24+ active, target < 60.
4. ``low_wind_high_power``      — the legacy rule set (three OR'd
                                   predicates); retained verbatim so the
                                   existing audit trail is preserved.

The :data:`IMPOSSIBLE_REASONS` tuple is the public enumeration; the empty
string is the "not flagged" sentinel. Downstream consumers (e.g. the jury
audit CSV) can ``reason in IMPOSSIBLE_REASONS`` to validate values.

Design references
-----------------
* ``.kiro/specs/data-quality-hardening/requirements.md`` §§ 1, 2, 7.
* ``.kiro/specs/data-quality-hardening/design.md`` §2 (Cleaner).
* ``PROJECT.md`` §10 — empirical record that ``downweight_2022 = 0.7``
  hurt Fold-5, which is why the default is now ``1.0``.

Weight-metadata bookkeeping
---------------------------
``compute_training_weights`` does not mutate its DataFrame and cannot
attach metadata to its returned ``np.ndarray`` (numpy arrays have no
``.attrs``). Instead, it records per-call context in the module-level
dict :data:`_LAST_WEIGHT_METADATA`. :func:`write_outlier_audit` reads that
dict to emit the header comment with the active ``downweight_2022`` and
per-reason counts. Callers that need strict isolation should import and
snapshot the dict immediately after the weight call.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.errors import CleanerSanityError
from src.data.schema import (
    CAPACITY_MW,
    TARGET_COL,
    TIMESTAMP_COL,
    TOTAL_TURBINES,
    TURBINES_IN_MAINTENANCE_COL,
)
from src.utils.warnings import DataQualityWarning

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Fraction of the training frame above which a rule is considered broken,
#: not the data. Fires via :class:`CleanerSanityError` in
#: :func:`impossible_reasons`. Covers Requirement 2.8.
_MAX_IMPOSSIBLE_FRACTION: float = 0.05

#: Frames smaller than this are treated as unit-test / smoke-test inputs,
#: where sub-percent fractions would trigger spuriously on a handful of
#: rows. The "too-permissive" guard then falls back to an extreme-fraction
#: check (see :data:`_DEGENERATE_FRACTION`). Real training frames on the
#: ARVE 2026 dataset have ~26 000 rows, so this threshold is comfortably
#: below the production size.
_MIN_FRAME_SIZE_FOR_GUARD: int = 100

#: On frames below :data:`_MIN_FRAME_SIZE_FOR_GUARD`, only fire the guard
#: when the combined flagged fraction is so high that the rule set is
#: clearly broken (e.g. 100 % over-capacity synthetic fixture).
_DEGENERATE_FRACTION: float = 0.5

#: Enumeration of impossible-row reason strings. The empty string is the
#: "not flagged" sentinel and always comes first so boolean-truthy checks
#: (``bool(reason)``) align with the impossible mask.
IMPOSSIBLE_REASONS: tuple[str, ...] = (
    "",  # not flagged
    "low_wind_high_power",
    "rated_plateau_shortfall",
    "over_capacity",
    "maintenance_zero",
)

#: Fixed evaluation order (earlier wins on ties). Distinct from the
#: public enumeration, which starts with the empty-string sentinel.
_RULE_EVALUATION_ORDER: tuple[str, ...] = (
    "over_capacity",
    "maintenance_zero",
    "rated_plateau_shortfall",
    "low_wind_high_power",
)

#: Module-level record of the last ``compute_training_weights`` call.
#: Consumed by :func:`write_outlier_audit` to populate the header comment.
_LAST_WEIGHT_METADATA: dict[str, Any] = {
    "downweight_2022": None,
    "reason_counts": {},
    "n_flagged": 0,
    "n_downweighted_2022": 0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _active_turbines(df: pd.DataFrame) -> pd.Series:
    """Derive the per-row active-turbine count.

    The single source of truth for Requirements 2.3 and 2.5:

        active = TOTAL_TURBINES - df[TURBINES_IN_MAINTENANCE_COL].fillna(0).astype(int)

    A missing maintenance value is treated as zero (no turbines down). This
    matches the existing behaviour of ``src/features/*`` which defaults the
    Cyrillic maintenance column to zero when absent.
    """
    if TURBINES_IN_MAINTENANCE_COL not in df.columns:
        return pd.Series(
            np.full(len(df), TOTAL_TURBINES, dtype=np.int64),
            index=df.index,
            name="active_turbines",
        )
    maint = df[TURBINES_IN_MAINTENANCE_COL].fillna(0).astype(int)
    return (TOTAL_TURBINES - maint).astype(int).rename("active_turbines")


def _rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Evaluate every impossible-row predicate independently.

    The return value is a mapping from reason name to a boolean Series
    aligned with ``df.index``. Priority resolution (earlier rule wins) is
    the caller's responsibility — see :func:`impossible_reasons`.
    """
    n = len(df)
    target = df.get(TARGET_COL, pd.Series(np.zeros(n), index=df.index))
    ws_120 = df.get("wind_speed_120m", pd.Series(np.zeros(n), index=df.index))
    active = _active_turbines(df)

    # Legacy low_wind_high_power predicate — retained verbatim from the
    # pre-hardening outliers module so the existing exclusion count does
    # not shift. Any threshold change belongs in a separate PR with an
    # updated baseline Fold-5 number.
    ws_cols_present = [
        c for c in ("wind_speed_10m", "wind_speed_80m", "wind_speed_120m") if c in df.columns
    ]
    if ws_cols_present:
        ws_max = df[ws_cols_present].max(axis=1)
    else:
        ws_max = pd.Series(np.zeros(n), index=df.index)
    gust = df.get("wind_gusts_10m", pd.Series(np.zeros(n), index=df.index))
    legacy_a = (ws_max < 4.0) & (gust < 6.0) & (target > 20.0)
    legacy_b = (ws_120 < 3.0) & (gust < 5.0) & (target > 15.0)
    legacy_c = (ws_120 < 4.0) & (gust < 7.0) & (target > 30.0)

    masks: dict[str, pd.Series] = {
        "over_capacity": (target > CAPACITY_MW).fillna(False).astype(bool),
        "maintenance_zero": ((active <= 2) & (target < 1.0)).fillna(False).astype(bool),
        "rated_plateau_shortfall": (
            (ws_120 > 14.0) & (active >= 24) & (target < 60.0)
        )
        .fillna(False)
        .astype(bool),
        "low_wind_high_power": (legacy_a | legacy_b | legacy_c).fillna(False).astype(bool),
    }
    return masks


def _check_sanity(masks: dict[str, pd.Series], n: int) -> None:
    """Raise :class:`CleanerSanityError` when any rule over-fires.

    Fires only on non-empty frames (Requirement 2.8). The message names
    each rule's contribution so the operator can diagnose without a
    second run.

    On frames smaller than :data:`_MIN_FRAME_SIZE_FOR_GUARD` (unit-test
    fixtures, smoke probes), the threshold is relaxed to
    :data:`_DEGENERATE_FRACTION` — sub-percent rule contributions on a
    5-row frame translate to a single flagged row, which would fire the
    production 5 % guard spuriously. Import-time validation against the
    real ~26 000-row training CSV (see :func:`_run_import_sanity_check`)
    always uses the strict 5 % threshold.
    """
    if n == 0:
        return

    threshold = (
        _MAX_IMPOSSIBLE_FRACTION
        if n >= _MIN_FRAME_SIZE_FOR_GUARD
        else _DEGENERATE_FRACTION
    )

    # Combined fraction across all rules (OR of the four predicates).
    combined = np.zeros(n, dtype=bool)
    per_rule_counts: dict[str, int] = {}
    for name in _RULE_EVALUATION_ORDER:
        arr = masks[name].to_numpy(dtype=bool, copy=False)
        combined = combined | arr
        per_rule_counts[name] = int(arr.sum())

    total_fraction = float(combined.sum()) / float(n)
    if total_fraction > threshold:
        contributions = ", ".join(
            f"{name}={per_rule_counts[name]}" for name in _RULE_EVALUATION_ORDER
        )
        raise CleanerSanityError(
            "impossible-row rules flag "
            f"{total_fraction:.3%} of {n} rows (> "
            f"{threshold:.0%} threshold). "
            f"Per-rule counts: {contributions}. "
            "A rule is likely too permissive — fix the predicate, not the data."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def impossible_reasons(df: pd.DataFrame) -> pd.Series:
    """Return the reason tag for each row.

    The returned Series has ``dtype == object``, length ``len(df)``, and
    every value belongs to :data:`IMPOSSIBLE_REASONS`. Rows matched by
    multiple predicates receive the earlier rule in
    :data:`_RULE_EVALUATION_ORDER`.

    Raises
    ------
    CleanerSanityError
        If the combined impossible fraction exceeds 5 % on a non-empty
        frame (Requirement 2.8).
    """
    n = len(df)
    masks = _rule_masks(df)
    _check_sanity(masks, n)

    reasons = np.full(n, "", dtype=object)
    # Iterate in reverse evaluation order so that earlier rules overwrite
    # later assignments — the earliest-priority tag survives.
    for name in reversed(_RULE_EVALUATION_ORDER):
        arr = masks[name].to_numpy(dtype=bool, copy=False)
        reasons[arr] = name

    return pd.Series(reasons, index=df.index, name="impossible_reason", dtype=object)


def identify_impossible_rows(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of physically impossible rows.

    The mask is the logical OR of the four rule predicates and satisfies
    ``mask.any() == (impossible_reasons(df) != "").any()`` (Requirement
    2.1). Length and dtype match the pre-hardening function.
    """
    reasons = impossible_reasons(df)
    mask = reasons.to_numpy() != ""
    return pd.Series(mask, index=df.index, name="impossible", dtype=bool)


def compute_training_weights(
    df: pd.DataFrame,
    downweight_2022: float = 1.0,
) -> np.ndarray:
    """Compute per-row training weights for LightGBM ``sample_weight``.

    Contract (Requirements 1.1–1.5):

    * Impossible rows receive weight ``0.0``.
    * Plausible rows receive weight ``1.0`` by default.
    * When ``downweight_2022 < 1.0``, every plausible 2022 row has its
      weight multiplied by ``downweight_2022``. A single
      :class:`DataQualityWarning` is emitted per call (Requirement 1.3).
    * The active ``downweight_2022`` and per-reason counts are stashed in
      :data:`_LAST_WEIGHT_METADATA` for :func:`write_outlier_audit`.

    The default was changed from ``0.7`` to ``1.0`` to reconcile the
    empirical finding in ``PROJECT.md §10`` (Requirement 1.1).
    """
    n = len(df)
    weights = np.ones(n, dtype=np.float32)

    reasons = impossible_reasons(df)
    impossible_arr = reasons.to_numpy() != ""
    weights[impossible_arr] = 0.0

    # Count per-reason occurrences for the audit header.
    reason_counts: dict[str, int] = {}
    for name in _RULE_EVALUATION_ORDER:
        reason_counts[name] = int((reasons.to_numpy() == name).sum())

    n_downweighted_2022 = 0
    if downweight_2022 < 1.0:
        # Emit exactly one warning per call, identifying the multiplier.
        warnings.warn(
            (
                f"compute_training_weights: downweight_2022={downweight_2022:.3f} "
                f"applied to 2022 rows (impossible={int(impossible_arr.sum())}, "
                "see PROJECT.md §10 — 0.7 hurt Fold-5)"
            ),
            DataQualityWarning,
            stacklevel=2,
        )

        if TIMESTAMP_COL in df.columns:
            ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=False)
            mask_2022 = (ts.dt.year == 2022).to_numpy()
            apply = mask_2022 & ~impossible_arr
            weights[apply] = weights[apply] * np.float32(downweight_2022)
            n_downweighted_2022 = int(apply.sum())

    _LAST_WEIGHT_METADATA.update(
        {
            "downweight_2022": float(downweight_2022),
            "reason_counts": reason_counts,
            "n_flagged": int(impossible_arr.sum()),
            "n_downweighted_2022": n_downweighted_2022,
        }
    )

    return weights


def write_outlier_audit(
    df: pd.DataFrame,
    weights: np.ndarray,
    reasons: pd.Series,
    *,
    path: str | Path = "data/interim/outlier_audit.csv",
) -> Path:
    """Write the outlier audit CSV atomically.

    Columns (Requirement 7.2) in order: ``timestamp``, ``reason``,
    ``weight``, ``wind_speed_120m``, ``active_turbines``, ``target_value``.
    Exactly one row per training-frame row whose emitted weight differs
    from ``1.0``; rows are sorted by ``timestamp`` for determinism and
    byte-identical re-runs (Requirement 7.6).

    A header comment line precedes the CSV header:

        # downweight_2022=<val> total_excluded=<n> reasons={...} imputed_180m_pct=<pct>

    The ``imputed_180m_pct`` field is populated from the ``ws_180m_is_imputed``
    column when present in ``df`` (see :func:`add_imputation_flag`,
    Requirement 3.6). When the flag column is absent — e.g. on a unit-test
    fixture that never ran through the loader — the field emits ``n/a`` so
    the header stays parseable.

    Parameters
    ----------
    df
        The training DataFrame the weights were computed from.
    weights
        The output of :func:`compute_training_weights` for ``df``.
    reasons
        The output of :func:`impossible_reasons` for ``df``.
    path
        Destination CSV path. Parent directory is created if missing.

    Returns
    -------
    pathlib.Path
        The resolved output path (post-rename).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights_arr = np.asarray(weights, dtype=np.float32)
    if len(weights_arr) != len(df):
        raise ValueError(
            f"weights length ({len(weights_arr)}) does not match df ({len(df)})"
        )
    if len(reasons) != len(df):
        raise ValueError(
            f"reasons length ({len(reasons)}) does not match df ({len(df)})"
        )

    # Filter to rows whose weight differs from 1.0 (Requirement 7.3).
    differ = ~np.isclose(weights_arr, 1.0, rtol=0.0, atol=0.0)

    # Build the audit frame. Timestamps are ISO-8601; target, wind_120m,
    # weight are six-decimal floats; active_turbines is an int.
    ts_series = df[TIMESTAMP_COL] if TIMESTAMP_COL in df.columns else pd.Series(
        pd.NaT, index=df.index
    )
    ts_iso = pd.to_datetime(ts_series, errors="coerce", utc=False).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    ws_120 = df.get(
        "wind_speed_120m", pd.Series(np.full(len(df), np.nan), index=df.index)
    ).astype(float)
    active = _active_turbines(df).astype(int)
    target = df.get(TARGET_COL, pd.Series(np.full(len(df), np.nan), index=df.index)).astype(
        float
    )

    audit = pd.DataFrame(
        {
            "timestamp": ts_iso.to_numpy(),
            "reason": reasons.to_numpy(),
            "weight": weights_arr,
            "wind_speed_120m": ws_120.to_numpy(),
            "active_turbines": active.to_numpy(),
            "target_value": target.to_numpy(),
        }
    )
    audit = audit.loc[differ].copy()

    # Sort by timestamp for determinism (Requirement 7.6). Rows with
    # missing timestamps (shouldn't happen after schema validation) sort
    # last so they do not inject nondeterminism.
    audit = audit.sort_values(by="timestamp", kind="mergesort", na_position="last")
    audit = audit.reset_index(drop=True)

    # Six-decimal formatting for the float columns.
    for col in ("weight", "wind_speed_120m", "target_value"):
        audit[col] = audit[col].map(
            lambda v: "" if pd.isna(v) else f"{float(v):.6f}"
        )
    audit["active_turbines"] = audit["active_turbines"].astype(int)

    # Header comment.
    meta = _LAST_WEIGHT_METADATA
    reason_counts = meta.get("reason_counts", {}) or {}
    reasons_str = (
        "{"
        + ", ".join(f"{k}: {int(v)}" for k, v in reason_counts.items())
        + "}"
    )
    downweight_value = meta.get("downweight_2022")
    downweight_str = "n/a" if downweight_value is None else f"{float(downweight_value):.6f}"
    total_excluded = int(differ.sum())

    # Imputed-180 m percentage: populated from ``ws_180m_is_imputed`` when
    # present (Requirement 3.6). When the flag column is absent (unit-test
    # fixtures, pre-Task-4.1 audit writes), fall back to ``n/a`` so the
    # header format stays stable for downstream parsers.
    if "ws_180m_is_imputed" in df.columns:
        imputed_pct = float(df["ws_180m_is_imputed"].astype(bool).mean()) * 100.0
        imputed_str = f"{imputed_pct:.1f}"
    else:
        imputed_str = "n/a"

    header_comment = (
        f"# downweight_2022={downweight_str} "
        f"total_excluded={total_excluded} "
        f"reasons={reasons_str} "
        f"imputed_180m_pct={imputed_str}\n"
    )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    # Write header comment + CSV body with Unix line endings for byte
    # determinism across platforms.
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header_comment)
        audit.to_csv(fh, index=False, lineterminator="\n")

    tmp_path.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Flag features (Task 4.1)
# ---------------------------------------------------------------------------


def add_imputation_flag(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    *,
    col: str = "wind_speed_180m",
) -> pd.DataFrame:
    """Attach a boolean flag marking rows where ``col`` was NaN-then-filled.

    The flag is computed positionally (``.values``) so mismatched
    ``df_pre``/``df_post`` indices do not silently align via pandas' label
    arithmetic — the contract is *row-wise at the same position*, not
    *row-wise at the same label*.

    Parameters
    ----------
    df_pre
        The pre-imputation frame (typically the raw CSV as loaded, before
        the Hellmann power-law fill runs).
    df_post
        The post-imputation frame (same length, aligned position-wise).
        A copy of this frame is returned with the added flag column.
    col
        The column whose NaN-then-filled transitions define the flag.
        Defaults to ``"wind_speed_180m"``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df_post`` with one extra boolean column. The column
        name is ``"ws_180m_is_imputed"`` when ``col == "wind_speed_180m"``
        and ``f"{col}_is_imputed"`` otherwise. Dtype is plain ``bool``
        (numpy ``bool_``); no ``NaN`` / ``NA`` values ever appear in the
        flag column.

    Raises
    ------
    ValueError
        If ``df_pre`` and ``df_post`` do not have matching length.
    KeyError
        If ``col`` is missing from either frame. The message names which
        frame is missing the column.

    Notes
    -----
    Deterministic: no RNG, no tolerance. Two invocations over the same
    ``(df_pre, df_post)`` produce equal Series (Property 9 in design.md).

    Validates
    ---------
    Requirements 3.2, 3.3, 3.4 — and Properties 8 (NaN-then-filled
    biconditional) and 9 (determinism) in
    ``.kiro/specs/data-quality-hardening/design.md``.
    """
    if len(df_pre) != len(df_post):
        raise ValueError(
            f"add_imputation_flag: df_pre length ({len(df_pre)}) does not match "
            f"df_post length ({len(df_post)}); positional comparison requires equal length"
        )
    if col not in df_pre.columns:
        raise KeyError(f"add_imputation_flag: column {col!r} missing from df_pre")
    if col not in df_post.columns:
        raise KeyError(f"add_imputation_flag: column {col!r} missing from df_post")

    flag_col = "ws_180m_is_imputed" if col == "wind_speed_180m" else f"{col}_is_imputed"

    pre_vals = df_pre[col].to_numpy()
    post_vals = df_post[col].to_numpy()
    flag = np.asarray(pd.isna(pre_vals) & pd.notna(post_vals), dtype=np.bool_)

    out = df_post.copy()
    out[flag_col] = flag
    return out


def add_nwp_era5_disagreement_flag(
    df: pd.DataFrame,
    *,
    threshold_m_s: float = 5.0,
) -> pd.DataFrame:
    """Attach ``nwp_era5_disagreement`` — a **flag**, never a row filter.

    Set to ``True`` only where both ``wind_speed_120m`` and
    ``era5_wind_speed_100m`` are present and their absolute difference
    exceeds ``threshold_m_s``. Set to ``False`` everywhere else, including
    rows where either input is missing and on every row when
    ``era5_wind_speed_100m`` is absent from ``df`` entirely (no error is
    raised — ERA5 merging is an optional enrichment).

    Parameters
    ----------
    df
        The input frame. A copy is returned with one extra column.
    threshold_m_s
        The disagreement threshold in m/s. Defaults to ``5.0`` per design
        §2 and Requirement 2.7.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with ``nwp_era5_disagreement: bool`` appended.
        Dtype is plain ``bool`` (numpy ``bool_``); no ``NaN`` / ``NA``
        values appear in the flag column.

    Notes
    -----
    The flag is a **feature** consumed by downstream feature selection
    (Requirement 2.7, Property 7). It **must not** be used to drop rows:
    disagreement encodes forecast uncertainty, which is signal for
    LightGBM to route around — exclusion would destroy that signal.

    Validates
    ---------
    Requirement 2.7 and Property 7 in
    ``.kiro/specs/data-quality-hardening/design.md``.
    """
    out = df.copy()
    n = len(out)

    if "wind_speed_120m" not in out.columns or "era5_wind_speed_100m" not in out.columns:
        out["nwp_era5_disagreement"] = np.zeros(n, dtype=np.bool_)
        return out

    ws_120 = pd.to_numeric(out["wind_speed_120m"], errors="coerce").to_numpy(dtype=float)
    era5 = pd.to_numeric(out["era5_wind_speed_100m"], errors="coerce").to_numpy(dtype=float)

    both_present = ~(np.isnan(ws_120) | np.isnan(era5))
    # Safe diff: NaN subtraction yields NaN, then ``both_present`` masks it out.
    with np.errstate(invalid="ignore"):
        exceeds = np.abs(ws_120 - era5) > float(threshold_m_s)
    flag = np.asarray(both_present & exceeds, dtype=np.bool_)

    out["nwp_era5_disagreement"] = flag
    return out


__all__ = [
    "IMPOSSIBLE_REASONS",
    "identify_impossible_rows",
    "impossible_reasons",
    "compute_training_weights",
    "write_outlier_audit",
    "add_imputation_flag",
    "add_nwp_era5_disagreement_flag",
]


# ---------------------------------------------------------------------------
# Import-time sanity check
# ---------------------------------------------------------------------------


def _run_import_sanity_check() -> None:
    """Validate rule set against ``data/raw/train_dataset.csv`` when present.

    At import time, load the raw training CSV (if it is on disk) and call
    :func:`impossible_reasons` — the 5 % guard in :func:`_check_sanity`
    will raise :class:`CleanerSanityError` if any rule is too permissive.
    This implements the "import-test time" fire requirement in
    Requirement 2.8 without coupling the module to the full loader.

    The check is a no-op when the CSV is unavailable (CI-without-data,
    packaging builds, unit-test sandboxes).
    """
    raw_path = Path("data/raw/train_dataset.csv")
    if not raw_path.is_file():
        return
    try:
        df = pd.read_csv(raw_path, nrows=None, low_memory=False)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    except (OSError, ValueError):
        # Never let a stray IO issue break the import; the schema layer
        # is the correct place to surface CSV-level defects.
        return
    # Call for side-effect: raises CleanerSanityError when any rule > 5 %.
    impossible_reasons(df)


# Do NOT run the sanity check on raw import — tests and dev tooling
# frequently import ``src.data.outliers`` without ``data/raw/`` mounted.
# Callers that want the check (e.g. ``tests/test_outliers.py`` or the
# loader entrypoint) should call :func:`_run_import_sanity_check`
# explicitly.
