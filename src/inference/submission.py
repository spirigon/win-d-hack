"""Submission CSV writer + format validation.

The organizer expects (updated format):
- CSV with header row: ``METEOFORECASTHOUR_OPENM_Datetime,<target_col>``
- 2126 data rows of (timestamp, prediction), total 2127 lines including header
- Decimal separator: dot.
- Values must be in [0, 90.09].

In addition to the format check performed by :func:`validate_submission`,
this module hosts the :class:`Sanity_Gate_Suite` described in
``ARCHITECTURE.md §8.3`` and Requirement 5 of the data-quality hardening
spec. The suite lives in :func:`_run_sanity_gates` — a private helper that
is invoked by :func:`write_submission` between the ``<output>.tmp`` write
and the atomic rename so a failing submission leaves the ``.tmp`` on disk
for post-mortem inspection.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.errors import SanityGateError
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.schemas import validate_submission as _schema_validate_submission
from src.utils.warnings import DataQualityWarning

#: Wind-speed bins for the monotonicity gate (Requirement 5.6). Half-open
#: on the right so ``12 m/s`` lands in the final ``[12, inf)`` bin. Order
#: matters — medians are compared in this sequence.
_WIND_BINS: tuple[tuple[float, float], ...] = (
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 10.0),
    (10.0, 12.0),
    (12.0, float("inf")),
)

#: Valid submission row counts (Requirement 5.2). ``2126`` is the Q1
#: forecast; ``24`` is the Day-i forecast.
_VALID_ROW_COUNTS: frozenset[int] = frozenset({2126, 24})


def validate_submission(preds: np.ndarray, expected_rows: int) -> None:
    """Raise if the predictions violate format rules."""
    if preds.shape[0] != expected_rows:
        raise ValueError(
            f"Row count mismatch: got {preds.shape[0]}, expected {expected_rows}"
        )
    if np.any(preds < 0):
        raise ValueError("Predictions contain negative values.")
    if np.any(preds > CAPACITY_MW + 1e-6):
        raise ValueError(f"Predictions exceed capacity ({CAPACITY_MW} MW).")
    if np.any(np.isnan(preds)):
        raise ValueError("Predictions contain NaN.")


def write_submission(
    preds: np.ndarray,
    output_path: str | Path,
    expected_rows: int,
    timestamps: np.ndarray | pd.Series | None = None,
    *,
    wind_speed_120m: np.ndarray | pd.Series | None = None,
) -> Path:
    """Clip, validate, and write the submission CSV atomically.

    New format: two-column CSV with header
    (``METEOFORECASTHOUR_OPENM_Datetime``, ``<target_col>``).

    If ``timestamps`` is provided, it must align with ``preds`` in submission
    row order. If omitted, the writer falls back to reading the original
    ``valid_features.csv`` to recover timestamps.

    Pipeline order (Task 8.2, Requirements 4.1 and 5.7):

    1. clip ``preds`` to ``[0, CAPACITY_MW]``;
    2. fast-fail in-memory shape/range/NaN check via
       :func:`validate_submission`;
    3. build the two-column ``out_df``;
    4. run the pandera :data:`src.data.schemas.SubmissionSchema` validator —
       raises :class:`src.data.schemas.SchemaError` **before** any disk
       write so the filesystem is untouched on schema failure;
    5. write ``<output>.tmp``;
    6. run :func:`_run_sanity_gates`; on :class:`SanityGateError` the
       ``.tmp`` stays on disk for diagnosis and the final path is never
       created;
    7. atomic ``tmp_path.replace(output_path)``.

    Parameters
    ----------
    wind_speed_120m:
        Optional paired wind-speed array aligned with ``preds``. When
        supplied, the paired sanity gates (``wind_sanity``,
        ``pct95_capacity``, ``wind_monotonicity``) run; otherwise a single
        :class:`DataQualityWarning` is emitted naming the skipped gates
        (Requirement 5.8).

    Returns the final path.
    """
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    validate_submission(preds, expected_rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if timestamps is None:
        # Fallback: read valid_features.csv to recover timestamp order.
        valid_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "valid_features.csv"
        ts_series = pd.read_csv(valid_path, usecols=[TIMESTAMP_COL])[TIMESTAMP_COL]
        if len(ts_series) != expected_rows:
            raise ValueError(
                f"valid_features.csv has {len(ts_series)} rows, expected {expected_rows}"
            )
        timestamps = ts_series.to_numpy()
    else:
        timestamps = np.asarray(timestamps)
        if len(timestamps) != expected_rows:
            raise ValueError(
                f"Timestamps length {len(timestamps)} != expected_rows {expected_rows}"
            )

    # Coerce timestamps to a pandas DatetimeIndex so SubmissionSchema's
    # ``pa.DateTime`` column type matches. ``pa.DateTime`` in pandera
    # 0.22.1 resolves to tz-naive ``datetime64[ns]`` — strip any tz on
    # the way in. The CSV bytes are unchanged because tz-naive
    # ``Timestamp`` objects also serialise to the same ISO-8601
    # representation used downstream.
    ts_idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts_idx.tz is not None:
        ts_idx = ts_idx.tz_convert("UTC").tz_localize(None)
    out_df = pd.DataFrame({
        TIMESTAMP_COL: ts_idx,
        TARGET_COL: preds,
    })

    # Pandera SubmissionSchema — raises SchemaError before any disk write.
    _schema_validate_submission(out_df)

    # Atomic write — ``.tmp`` first, then gate, then rename.
    tmp_path = output_path.with_suffix(".tmp")
    out_df.to_csv(tmp_path, index=False, float_format="%.6f", encoding="utf-8")

    # Sanity gates — raises SanityGateError on failure; .tmp stays on disk
    # for post-mortem inspection because the rename is gated behind this.
    _run_sanity_gates(
        np.asarray(preds),
        np.asarray(timestamps),
        expected_rows=expected_rows,
        wind_speed_120m=(
            None if wind_speed_120m is None else np.asarray(wind_speed_120m)
        ),
    )

    tmp_path.replace(output_path)
    total_lines = len(out_df) + 1  # +1 for header
    print(f"  Submission written: {output_path} ({total_lines} lines, {len(preds)} predictions)")
    return output_path


# --------------------------------------------------------------------------- #
# Sanity_Gate_Suite (Requirement 5)
# --------------------------------------------------------------------------- #

def _format_offending(indices: np.ndarray, *, limit: int = 3) -> str:
    """Return a compact ``"N rows violated (indices: [...])"`` string.

    The first ``limit`` row indices are listed; larger failure sets get a
    ``, ...`` suffix so the error message stays readable regardless of how
    many rows are bad.
    """
    total = int(indices.size)
    head = indices[:limit].tolist()
    suffix = ", ..." if total > limit else ""
    return f"{total} rows violated (indices: {head}{suffix})"


def _check_range(preds: np.ndarray) -> None:
    """Gate R5.1: every element of ``preds`` lies in ``[0, CAPACITY_MW]``.

    NaNs count as violations — a NaN is neither ``>= 0`` nor
    ``<= CAPACITY_MW``. Using ``~(lo & hi)`` propagates NaNs into the
    violation set without a dedicated NaN branch.
    """
    lo = preds >= 0.0
    hi = preds <= CAPACITY_MW
    bad = np.flatnonzero(~(lo & hi))
    if bad.size:
        raise SanityGateError(
            f"range: {_format_offending(bad)} "
            f"(allowed: [0.0, {CAPACITY_MW}])"
        )


def _check_row_count(preds: np.ndarray, *, expected_rows: int) -> None:
    """Gate R5.2: ``len(preds) == expected_rows`` and the count is in
    ``{2126, 24}``.

    Two failure modes are reported separately so the jury note can cite
    which one fired — an off-by-one rarely has the same fix as a totally
    wrong ``expected_rows`` argument.
    """
    if expected_rows not in _VALID_ROW_COUNTS:
        raise SanityGateError(
            f"row_count: expected_rows={expected_rows} not in "
            f"{sorted(_VALID_ROW_COUNTS)}"
        )
    if preds.shape[0] != expected_rows:
        raise SanityGateError(
            f"row_count: got {preds.shape[0]} predictions, expected "
            f"{expected_rows}"
        )


def _check_duplicate_timestamp(timestamps: np.ndarray) -> None:
    """Gate R5.3: no timestamp appears twice."""
    # ``pd.unique`` preserves dtype (including numpy.datetime64) without a
    # pandas Index round-trip and is faster than ``np.unique`` on small N.
    if pd.unique(timestamps).shape[0] != timestamps.shape[0]:
        # Surface the first duplicate indices so the caller can diagnose
        # the ``.tmp`` left on disk.
        seen: dict[object, int] = {}
        dup_indices: list[int] = []
        for i, t in enumerate(timestamps):
            # ``np.datetime64`` instances are hashable; raw Python objects
            # work too. Guard against type mismatches by stringifying.
            key = t if not isinstance(t, np.datetime64) else t.astype("datetime64[ns]")
            if key in seen:
                dup_indices.append(i)
            else:
                seen[key] = i
        raise SanityGateError(
            f"duplicate_timestamp: "
            f"{_format_offending(np.asarray(dup_indices))}"
        )


def _check_wind_sanity(preds: np.ndarray, wind_speed_120m: np.ndarray) -> None:
    """Gate R5.4 — per-row, not pair-reduced.

    ``ws < 1.0`` must imply ``pred < 2.0`` and ``ws > 25.0`` must imply
    ``pred < 5.0``. Both checks collapse into one violation vector so the
    first offending index reflects the overall earliest breach.
    """
    low_wind = wind_speed_120m < 1.0
    high_wind = wind_speed_120m > 25.0
    low_wind_fail = low_wind & (preds >= 2.0)
    high_wind_fail = high_wind & (preds >= 5.0)
    bad = np.flatnonzero(low_wind_fail | high_wind_fail)
    if bad.size:
        raise SanityGateError(
            f"wind_sanity: {_format_offending(bad)} "
            f"(ws<1 → pred<2.0, ws>25 → pred<5.0)"
        )


def _check_pct95_capacity(preds: np.ndarray) -> None:
    """Gate R5.5: ``np.percentile(preds, 95) <= CAPACITY_MW``.

    Under linear interpolation, a 95th percentile can exceed
    ``CAPACITY_MW`` even when no single value does (e.g. if the sample
    straddles the boundary). This gate is therefore independent of the
    range gate.
    """
    pct95 = float(np.percentile(preds, 95))
    if pct95 > CAPACITY_MW:
        raise SanityGateError(
            f"pct95_capacity: 95th percentile {pct95:.4f} > "
            f"{CAPACITY_MW} (CAPACITY_MW)"
        )


def _bin_index(ws: np.ndarray) -> np.ndarray:
    """Assign each ``ws`` element to an index into ``_WIND_BINS`` or ``-1``
    if it falls outside the observed wind window ``[4, inf)``.

    Values ``< 4`` return ``-1`` (not covered by the monotonicity gate).
    """
    idx = np.full(ws.shape, -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(_WIND_BINS):
        mask = (ws >= lo) & (ws < hi)
        idx[mask] = i
    return idx


def _check_wind_monotonicity(
    preds: np.ndarray, wind_speed_120m: np.ndarray
) -> None:
    """Gate R5.6: per-bin median predictions non-decreasing over
    ``[4, 6), [6, 8), [8, 10), [10, 12), [12, ∞)``.

    Empty bins are skipped (not treated as zero): if the medians of the
    populated bins, read left-to-right, are non-decreasing, the gate
    passes. A bin with a single observation uses that observation as its
    median.
    """
    bins = _bin_index(wind_speed_120m)
    populated: list[tuple[int, float]] = []
    for i in range(len(_WIND_BINS)):
        mask = bins == i
        if not mask.any():
            continue
        populated.append((i, float(np.median(preds[mask]))))
    # Non-decreasing check with the first violating pair named.
    for k in range(1, len(populated)):
        prev_i, prev_m = populated[k - 1]
        cur_i, cur_m = populated[k]
        if cur_m < prev_m:
            raise SanityGateError(
                f"wind_monotonicity: median dropped from bin "
                f"{_WIND_BINS[prev_i]} ({prev_m:.4f}) to bin "
                f"{_WIND_BINS[cur_i]} ({cur_m:.4f})"
            )


def _run_sanity_gates(
    preds: np.ndarray,
    timestamps: np.ndarray,
    *,
    expected_rows: int,
    wind_speed_120m: np.ndarray | None = None,
) -> None:
    """Enforce the six Sanity_Gate_Suite checks from Requirement 5.

    Unpaired gates (always run):

    - ``range``: every prediction lies in ``[0, CAPACITY_MW]`` (R5.1).
    - ``row_count``: ``len(preds) == expected_rows`` and the count is in
      ``{2126, 24}`` (R5.2).
    - ``duplicate_timestamp``: ``len(unique(timestamps)) == len(timestamps)``
      (R5.3).

    Paired gates (run only when ``wind_speed_120m is not None``):

    - ``wind_sanity``: per-row — ``ws < 1`` ⇒ ``pred < 2``; ``ws > 25`` ⇒
      ``pred < 5`` (R5.4).
    - ``pct95_capacity``: ``np.percentile(preds, 95) <= CAPACITY_MW``
      (R5.5).
    - ``wind_monotonicity``: per-bin median predictions non-decreasing on
      bins ``[4, 6), [6, 8), [8, 10), [10, 12), [12, ∞)`` (R5.6); empty
      bins are skipped, not treated as zero.

    When ``wind_speed_120m is None``, emits exactly one
    :class:`DataQualityWarning` naming the skipped paired gates (R5.8).

    Raises :class:`SanityGateError` naming the failing gate and the first
    few offending row indices. The caller is responsible for the atomic
    ``.tmp → rename``; this helper does not touch disk.
    """
    preds = np.asarray(preds)
    timestamps = np.asarray(timestamps)

    # Unpaired gates — always run, in increasing cost order.
    _check_row_count(preds, expected_rows=expected_rows)
    _check_duplicate_timestamp(timestamps)
    _check_range(preds)

    if wind_speed_120m is None:
        warnings.warn(
            "Paired sanity gates skipped: wind_sanity, pct95_capacity, "
            "wind_monotonicity (wind_speed_120m not supplied)",
            DataQualityWarning,
            stacklevel=2,
        )
        return

    ws = np.asarray(wind_speed_120m)
    if ws.shape[0] != preds.shape[0]:
        raise SanityGateError(
            f"wind_sanity: wind_speed_120m length {ws.shape[0]} "
            f"does not match predictions length {preds.shape[0]}"
        )

    _check_wind_sanity(preds, ws)
    _check_pct95_capacity(preds)
    _check_wind_monotonicity(preds, ws)
