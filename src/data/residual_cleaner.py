"""Opt-in residual-based label-noise second-pass cleaner.

This module is the **Residual_Cleaner** half of the Data Quality Hardening
pack (spec: ``.kiro/specs/data-quality-hardening``) — see design §3. It
looks at out-of-fold (OOF) predictions for the training set and flags rows
whose residual ``r_i = y_i − y_oof_i`` is unusually large relative to the
per-``wind_speed_120m``-bin trimmed standard deviation. The flagged rows
are label-noise *candidates*, not guaranteed defects; downstream training
code multiplies their sample weight by ``0.0`` only when the caller
explicitly passes ``--residual-clean`` on the training CLI.

The cleaner is **inert on the default code path** (Requirement 6.7): it is
never imported or invoked from ``src/training/`` unless the opt-in flag is
set (Task 7.2). When it does run, it enforces a hard 2 %-cap
(:data:`_MAX_FLAGGED_FRACTION` — Requirement 6.5) and raises
:class:`CleanerSanityError` if the ``k``-threshold would flag more of the
training set than the contract allows.

Design highlights
-----------------
* **Deterministic.** No RNG on the hot path. Bin edges, sorted trim, and
  per-bin std are all deterministic functions of the aligned arrays.
* **Atomic audit write.** ``data/interim/residual_flags_{fold_name}.csv``
  is written via ``.tmp → rename`` so a crashed run cannot leave a
  partially-written audit. Header-only when no rows are flagged
  (Requirement 6.4).
* **Left-join alignment.** OOF is aligned by timestamp via left-merge. A
  coverage check raises if fewer than 50 % of ``train_df`` timestamps have
  a matching OOF entry — partial OOFs are fine, empty OOFs are not.
* **First-order trimmed std.** The per-bin ``sigma_bin`` is the trimmed
  std of all residuals in the bin (top and bottom 5 % trimmed). The exact
  leave-one-out refinement in design §3 step 5 is a follow-up; the test
  battery (Tasks 7.3–7.7) catches any drift from the simpler form.

Public surface: :func:`flag_residuals`.

The module does **not** re-declare :class:`CleanerSanityError` — it is
imported from :mod:`src.data.errors` to keep the exception identity
shared with :mod:`src.data.outliers`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.errors import CleanerSanityError
from src.data.schema import TARGET_COL, TIMESTAMP_COL

__all__ = ["flag_residuals"]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Hard cap on the fraction of aligned rows that may be flagged
#: (Requirement 6.5). Breaching this fires :class:`CleanerSanityError`.
_MAX_FLAGGED_FRACTION: float = 0.02

#: ``ws_bin`` is clipped to this closed integer range so a single
#: high-wind observation in a degenerate tail bin does not produce a
#: zero-std threshold. 30 m/s at 120 m is well past the Siemens Gamesa
#: SG 3.4-132 cut-out, so clipping has no effect on normal data.
_WS_BIN_CLIP_LOW: int = 0
_WS_BIN_CLIP_HIGH: int = 30

#: Minimum OOF-coverage fraction below which the alignment is judged
#: unsafe (partial OOFs are acceptable; near-empty OOFs are not).
_MIN_OOF_COVERAGE_FRACTION: float = 0.5

#: Fraction of each bin's residuals trimmed from the top and the bottom
#: before computing the per-bin std.
_TRIM_FRACTION: float = 0.05

#: Sentinel default for the ``audit_path`` keyword. When the caller leaves
#: it at this value, :func:`flag_residuals` substitutes the
#: ``residual_flags_{fold_name}.csv`` template; otherwise the explicit
#: path is honoured. Using a sentinel rather than ``None`` preserves the
#: ``str | Path`` type annotation in the design signature.
_DEFAULT_AUDIT_PATH: Path = Path("data/interim/residual_flags.csv")

#: Fixed audit column order (Requirement 6.4).
_AUDIT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "y",
    "y_oof",
    "residual",
    "ws_120m",
    "ws_bin",
    "threshold",
    "flagged",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trimmed_std(values: np.ndarray, trim: float = _TRIM_FRACTION) -> float:
    """Deterministic per-bin trimmed standard deviation.

    Trims the top and bottom ``trim`` fraction of values before computing
    the population standard deviation (``ddof=0``). Returns ``np.inf`` when

    * the bin has fewer than three rows (too few to form a reliable
      threshold), or
    * the trimmed std is degenerate (zero or non-finite).

    Returning ``np.inf`` effectively disables flagging on that bin: no
    row can exceed ``k * inf``. This is the conservative choice — a
    degenerate bin is not evidence of an outlier.

    The design allows a first-order per-bin trimmed std (rather than the
    exact leave-one-out formula) as the initial implementation; see
    design §3 step 5 for the LOO refinement should future determinism
    tests demand it.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 3:
        return float("inf")
    sorted_vals = np.sort(arr)
    lo = int(np.floor(trim * n))
    hi = n - int(np.floor(trim * n))
    if hi <= lo + 1:
        return float("inf")
    trimmed = sorted_vals[lo:hi]
    s = float(np.std(trimmed, ddof=0))
    if s <= 0.0 or not np.isfinite(s):
        return float("inf")
    return s


def _empty_audit_frame() -> pd.DataFrame:
    """Return an empty DataFrame with exactly the audit-column schema."""
    return pd.DataFrame({col: [] for col in _AUDIT_COLUMNS})


def _write_audit_atomic(path: Path, df: pd.DataFrame) -> None:
    """Write the audit CSV atomically via ``.tmp → rename``.

    Numeric columns are formatted to six decimals so two runs on the same
    input produce byte-identical files; booleans render as ``True`` /
    ``False``; timestamps render as ISO-8601 seconds
    (``YYYY-MM-DDTHH:MM:SS``). An empty frame produces a header-only file
    (Requirement 6.4). The writer uses Unix line endings unconditionally
    to avoid cross-platform byte drift.
    """
    # Reindex to the canonical column order; fill any missing columns
    # with empty values so we never emit a column in the wrong position.
    ordered = pd.DataFrame(
        {col: (df[col] if col in df.columns else []) for col in _AUDIT_COLUMNS}
    )

    if not ordered.empty:
        ts_series = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=False)
        ordered["timestamp"] = ts_series.dt.strftime("%Y-%m-%dT%H:%M:%S")
        for c in ("y", "y_oof", "residual", "ws_120m", "threshold"):
            ordered[c] = ordered[c].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.6f}"
            )
        ordered["ws_bin"] = ordered["ws_bin"].astype(int)
        ordered["flagged"] = ordered["flagged"].astype(bool).map(
            lambda b: "True" if b else "False"
        )

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
        ordered.to_csv(fh, index=False, lineterminator="\n")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flag_residuals(
    train_df: pd.DataFrame,
    oof_path: str | Path,
    *,
    k: float = 3.0,
    ws_bin_width: float = 1.0,
    audit_path: str | Path = _DEFAULT_AUDIT_PATH,
    fold_name: str = "all",
) -> pd.Series:
    """Flag rows whose OOF residual exceeds ``k × sigma_bin``.

    Parameters
    ----------
    train_df
        Training DataFrame. Must contain :data:`~src.data.schema.TIMESTAMP_COL`,
        :data:`~src.data.schema.TARGET_COL`, and ``wind_speed_120m``.
    oof_path
        Path to a parquet file with columns :data:`~src.data.schema.TIMESTAMP_COL`
        and ``y_oof`` — the out-of-fold predictions for the same window.
    k
        Threshold multiplier (default ``3.0``). A row is flagged when
        ``|r_i| > k * sigma_bin[ws_bin_i]``.
    ws_bin_width
        Bin width in m/s (default ``1.0``). ``ws_bin = floor(ws_120m /
        ws_bin_width)`` clipped to ``[0, 30]``.
    audit_path
        Destination CSV path. When left at the default sentinel, the
        cleaner substitutes ``data/interim/residual_flags_{fold_name}.csv``.
        Explicit paths are honoured verbatim.
    fold_name
        Used in the default audit path template. Ignored when the caller
        supplies an explicit ``audit_path``.

    Returns
    -------
    pandas.Series[bool]
        Boolean Series aligned on ``train_df.index``. Rows without a
        matching OOF entry default to ``False``.

    Raises
    ------
    CleanerSanityError
        If the flagged fraction exceeds :data:`_MAX_FLAGGED_FRACTION`
        (2 %). The audit CSV is written **before** the raise so postmortem
        diagnosis does not require a re-run.
    ValueError
        If the OOF parquet lacks required columns, or if fewer than 50 %
        of ``train_df`` timestamps have a matching OOF entry, or if
        ``wind_speed_120m`` contains non-finite values after alignment.
    """
    # ------------------------------------------------------------------
    # 1. Resolve audit_path. The sentinel default means "use fold_name
    #    template"; anything else is honoured verbatim.
    # ------------------------------------------------------------------
    audit_p = Path(audit_path)
    if audit_p == _DEFAULT_AUDIT_PATH:
        audit_p = Path(f"data/interim/residual_flags_{fold_name}.csv")
    audit_p.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Validate required columns on the training frame.
    # ------------------------------------------------------------------
    for col in (TIMESTAMP_COL, TARGET_COL, "wind_speed_120m"):
        if col not in train_df.columns:
            raise ValueError(f"train_df is missing required column {col!r}")

    n_train = len(train_df)
    flagged_mask = np.zeros(n_train, dtype=bool)

    # Empty training frame: write header-only audit and return. No 2 %-cap
    # check on empty frames (Requirement 6.5 reads "would flag more than
    # 2 %" — a 0/0 ratio is not "more than").
    if n_train == 0:
        _write_audit_atomic(audit_p, _empty_audit_frame())
        return pd.Series(
            flagged_mask, index=train_df.index, name="residual_flag", dtype=bool
        )

    # ------------------------------------------------------------------
    # 3. Load OOF parquet; validate its columns; dedupe timestamps.
    # ------------------------------------------------------------------
    oof_df = pd.read_parquet(oof_path)
    if TIMESTAMP_COL not in oof_df.columns:
        raise ValueError(
            f"OOF parquet is missing timestamp column {TIMESTAMP_COL!r}"
        )
    if "y_oof" not in oof_df.columns:
        raise ValueError("OOF parquet is missing 'y_oof' column")

    oof_slim = (
        oof_df.loc[:, [TIMESTAMP_COL, "y_oof"]]
        .drop_duplicates(subset=TIMESTAMP_COL, keep="last")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 4. Left-merge preserves left-row order. indicator marks coverage.
    # ------------------------------------------------------------------
    merged = train_df.merge(
        oof_slim, on=TIMESTAMP_COL, how="left", indicator="_oof_marker"
    )
    aligned_mask_pos = (merged["_oof_marker"] == "both").to_numpy()
    n_matched = int(aligned_mask_pos.sum())

    if n_matched / n_train < _MIN_OOF_COVERAGE_FRACTION:
        raise ValueError(
            f"OOF coverage {n_matched}/{n_train} "
            f"(< {_MIN_OOF_COVERAGE_FRACTION:.0%}) is too low for residual "
            "cleaning; check the OOF parquet timestamp alignment."
        )

    # No aligned rows: write header-only and return. Only reachable if
    # _MIN_OOF_COVERAGE_FRACTION is loosened in the future; kept as an
    # explicit short-circuit for clarity.
    if n_matched == 0:
        _write_audit_atomic(audit_p, _empty_audit_frame())
        return pd.Series(
            flagged_mask, index=train_df.index, name="residual_flag", dtype=bool
        )

    # ------------------------------------------------------------------
    # 5. Extract aligned arrays.
    # ------------------------------------------------------------------
    merged_sub = merged.loc[aligned_mask_pos].reset_index(drop=True)
    ts_aligned = merged_sub[TIMESTAMP_COL].to_numpy()
    y = merged_sub[TARGET_COL].astype(float).to_numpy()
    y_oof = merged_sub["y_oof"].astype(float).to_numpy()
    ws_120m = merged_sub["wind_speed_120m"].astype(float).to_numpy()

    if not np.all(np.isfinite(ws_120m)):
        raise ValueError(
            "wind_speed_120m contains NaN/inf after OOF alignment; "
            "run schema validation upstream."
        )

    residual = y - y_oof

    # ------------------------------------------------------------------
    # 6. Binning (deterministic, no RNG).
    # ------------------------------------------------------------------
    ws_bin = np.clip(
        np.floor(ws_120m / ws_bin_width).astype(np.int64),
        _WS_BIN_CLIP_LOW,
        _WS_BIN_CLIP_HIGH,
    )

    # Per-bin trimmed std. groupby(sort=True) is deterministic.
    bin_frame = pd.DataFrame({"bin": ws_bin, "r": residual})
    sigma_by_bin: dict[int, float] = {}
    for bin_id, grp in bin_frame.groupby("bin", sort=True):
        sigma_by_bin[int(bin_id)] = _trimmed_std(
            grp["r"].to_numpy(), trim=_TRIM_FRACTION
        )

    sigma_vec = np.fromiter(
        (sigma_by_bin[int(b)] for b in ws_bin),
        dtype=np.float64,
        count=ws_bin.size,
    )
    threshold = k * sigma_vec

    # Residuals that are NaN compare False under `>`, which is the desired
    # behaviour (do not flag missing-residual rows).
    flagged_aligned = np.abs(residual) > threshold

    flagged_count = int(flagged_aligned.sum())
    flagged_fraction = flagged_count / n_matched

    # ------------------------------------------------------------------
    # 7. Write the audit FIRST (so postmortem works even on the cap raise).
    # ------------------------------------------------------------------
    if flagged_count == 0:
        _write_audit_atomic(audit_p, _empty_audit_frame())
    else:
        audit_df = pd.DataFrame(
            {
                "timestamp": ts_aligned,
                "y": y,
                "y_oof": y_oof,
                "residual": residual,
                "ws_120m": ws_120m,
                "ws_bin": ws_bin,
                "threshold": threshold,
                "flagged": flagged_aligned,
            }
        )
        audit_flagged = (
            audit_df[audit_df["flagged"]]
            .sort_values(by="timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        _write_audit_atomic(audit_p, audit_flagged)

    # ------------------------------------------------------------------
    # 8. Enforce the 2 %-cap (after the audit is on disk).
    # ------------------------------------------------------------------
    if flagged_fraction > _MAX_FLAGGED_FRACTION:
        raise CleanerSanityError(
            f"residual cleaner flags {flagged_fraction:.3%} "
            f"({flagged_count}/{n_matched}) of aligned rows, "
            f"above the {_MAX_FLAGGED_FRACTION:.0%} cap with k={k}. "
            "Tune `k` upward, inspect the OOF parquet, or re-train."
        )

    # ------------------------------------------------------------------
    # 9. Project flagged_aligned back onto train_df.index.
    # ------------------------------------------------------------------
    flagged_mask[aligned_mask_pos] = flagged_aligned
    return pd.Series(
        flagged_mask, index=train_df.index, name="residual_flag", dtype=bool
    )
