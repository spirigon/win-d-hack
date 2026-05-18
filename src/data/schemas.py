"""Pandera schemas for the data-quality hardening pack.

This module sits at the Loader boundary. It defines four schemas —
``GenerationSchema``, ``WeatherSchema``, ``ForecastSchema``,
``SubmissionSchema`` — and the four ``validate_*`` helpers that run them
with ``lazy=True``, classify the collected violations, and persist a
report at ``data/interim/schema_report.json`` via an atomic
``.tmp → rename`` write.

Classification contract (Requirements 4.5 / 4.6):

* **Fatal** (raise :class:`SchemaError`):
    - duplicate ``TIMESTAMP_COL``
    - any ``TARGET_COL`` value ``> CAPACITY_MW``
    - ``TARGET_COL`` column missing / renamed away from the Cyrillic literal
* **Warning** (``warnings.warn(..., DataQualityWarning)``, write report,
  continue): any Weather / Forecast range deviation that does not have a
  target-side effect; one warning per violation *class* (column × check),
  not per row.

Note
----
The existing ``src/data/schema.py`` (no trailing ``s``) keeps holding bare
column-name constants. This file imports from it — do not duplicate those
constants here.

``pandera==0.22.1`` exposes its pandas API at the root ``pandera``
namespace (the ``pandera.pandas`` sub-accessor lands in the 0.23 line).
We import as ``pa`` so call sites match the design doc byte-for-byte.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
import pandera.errors as pa_errors

from src.data.schema import (
    CAPACITY_MW,
    TARGET_COL,
    TIMESTAMP_COL,
    TURBINES_IN_MAINTENANCE_COL,
)
from src.utils.warnings import DataQualityWarning

__all__ = [
    "SchemaError",
    "GenerationSchema",
    "WeatherSchema",
    "ForecastSchema",
    "SubmissionSchema",
    "validate_generation",
    "validate_weather",
    "validate_forecast",
    "validate_submission",
    "SCHEMA_REPORT_PATH",
    "MAX_FAILURE_CASES",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Capped number of failure-case rows persisted in the report JSON
#: (Requirement 4.4).
MAX_FAILURE_CASES: int = 1000

#: Destination of the lazy-validation report. Overwritten atomically on
#: every ``validate_*`` call (Requirement 4.4).
SCHEMA_REPORT_PATH: Path = Path("data/interim/schema_report.json")

#: Expected submission row counts: 2126 for the Q1 forecast, 24 for the
#: Day-i forecast (``ARCHITECTURE.md §4.1``, Requirement 5.2).
_SUBMISSION_ROW_COUNTS: frozenset[int] = frozenset({2126, 24})

#: Weather range limits (m/s, °C); see design §1 "WeatherSchema".
_WS_MIN: float = 0.0
_WS_MAX: float = 60.0
_TEMP_MIN: float = -40.0
_TEMP_MAX: float = 50.0


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class SchemaError(Exception):
    """Raised on fatal pandera violations.

    Three fatal classes exist (Requirement 4.5):

    * duplicate ``TIMESTAMP_COL``
    * ``TARGET_COL`` value above ``CAPACITY_MW``
    * ``TARGET_COL`` missing / renamed

    Other range-class violations are warnings, not errors (Requirement
    4.6). The exception is raised *after* the report JSON has been
    written so the jury note can cite the exact failure.
    """


# --------------------------------------------------------------------------- #
# Dataframe-level checks
# --------------------------------------------------------------------------- #

def _gust_ge_ws_10m(df: pd.DataFrame) -> bool:
    """Physical plausibility: a gust cannot be lower than the sustained wind.

    NaNs on either side are treated as "not applicable" (the check
    cannot fire on missing data). Returning ``False`` when all comparable
    rows satisfy the relation triggers a pandera dataframe-level
    failure that the caller will emit as a warning.
    """
    gust = df["wind_gusts_10m"]
    ws = df["wind_speed_10m"]
    mask = gust.notna() & ws.notna()
    if not mask.any():
        return True
    return bool((gust[mask] >= ws[mask]).all())


def _submission_row_count(df: pd.DataFrame) -> bool:
    """Enforce ``len(df) ∈ {2126, 24}`` at the SubmissionSchema level."""
    return len(df) in _SUBMISSION_ROW_COUNTS


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

#: Training-frame schema. The Cyrillic ``TARGET_COL`` and
#: ``TURBINES_IN_MAINTENANCE_COL`` literals are preserved byte-for-byte
#: (Requirement 4.7).
GenerationSchema: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        TIMESTAMP_COL: pa.Column(
            pa.DateTime,
            unique=True,
            nullable=False,
        ),
        TARGET_COL: pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(0.0, CAPACITY_MW, include_max=True),
        ),
        TURBINES_IN_MAINTENANCE_COL: pa.Column(
            pa.Int,
            nullable=False,
            checks=pa.Check.in_range(0, 26, include_max=True),
        ),
    },
    strict="filter",
    ordered=False,
)


def _wind_column(*, nullable: bool = False) -> pa.Column:
    return pa.Column(
        pa.Float,
        nullable=nullable,
        checks=pa.Check.in_range(_WS_MIN, _WS_MAX, include_max=True),
    )


#: Weather-feature schema. ``wind_speed_180m`` is the imputed channel and
#: is allowed to be nullable pre-Hellmann (design §1). The dataframe-level
#: gust ≥ ws_10m check is a soft physical plausibility signal — a failure
#: is classified as a warning, never fatal.
WeatherSchema: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        "wind_speed_10m": _wind_column(nullable=False),
        "wind_speed_80m": _wind_column(nullable=False),
        "wind_speed_120m": _wind_column(nullable=False),
        "wind_speed_180m": _wind_column(nullable=True),
        "wind_gusts_10m": pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(_WS_MIN, _WS_MAX, include_max=True),
        ),
        "temperature_2m": pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(_TEMP_MIN, _TEMP_MAX, include_max=True),
        ),
    },
    checks=pa.Check(
        _gust_ge_ws_10m,
        name="gust_ge_ws_10m",
        element_wise=False,
    ),
    strict="filter",
    ordered=False,
)


#: Forecast-feature schema. Same columns as Weather plus an optional
#: ``lead_hour`` — present on hour-indexed forecast frames, absent on
#: pre-merged ones.
ForecastSchema: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        "wind_speed_10m": _wind_column(nullable=False),
        "wind_speed_80m": _wind_column(nullable=False),
        "wind_speed_120m": _wind_column(nullable=False),
        "wind_speed_180m": _wind_column(nullable=True),
        "wind_gusts_10m": pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(_WS_MIN, _WS_MAX, include_max=True),
        ),
        "temperature_2m": pa.Column(
            pa.Float,
            nullable=True,
            checks=pa.Check.in_range(_TEMP_MIN, _TEMP_MAX, include_max=True),
        ),
        "lead_hour": pa.Column(pa.Int, required=False, nullable=False),
    },
    checks=pa.Check(
        _gust_ge_ws_10m,
        name="gust_ge_ws_10m",
        element_wise=False,
    ),
    strict="filter",
    ordered=False,
)


#: Submission schema. Matches what ``src/inference/submission.py``
#: actually writes: two columns — ``TIMESTAMP_COL`` (unique) and the
#: Cyrillic ``TARGET_COL`` in ``[0, CAPACITY_MW]``. Row count ∈ {2126,
#: 24} is enforced at the dataframe level.
SubmissionSchema: pa.DataFrameSchema = pa.DataFrameSchema(
    columns={
        TIMESTAMP_COL: pa.Column(
            pa.DateTime,
            unique=True,
            nullable=False,
        ),
        TARGET_COL: pa.Column(
            pa.Float,
            nullable=False,
            checks=pa.Check.in_range(0.0, CAPACITY_MW, include_max=True),
        ),
    },
    checks=pa.Check(
        _submission_row_count,
        name="row_count_in_{2126,24}",
        element_wise=False,
    ),
    strict="filter",
    ordered=False,
)


# --------------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------------- #

def _now_iso_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _failure_cases_to_records(
    failure_cases: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Convert pandera's ``failure_cases`` frame into a capped record list.

    Returns ``(records, truncated)`` where ``truncated`` is ``True`` iff
    the original frame had more than :data:`MAX_FAILURE_CASES` rows
    (Requirement 4.4).
    """
    if failure_cases is None or len(failure_cases) == 0:
        return [], False
    total = len(failure_cases)
    truncated = total > MAX_FAILURE_CASES
    head = failure_cases.head(MAX_FAILURE_CASES)
    # ``to_dict(orient="records")`` preserves the six pandera columns:
    # schema_context, column, check, check_number, failure_case, index.
    return head.to_dict(orient="records"), truncated


def _write_report_atomic(
    *,
    path: Path,
    source_path: str,
    rows: int,
    status: str,
    failure_cases: pd.DataFrame | None,
) -> None:
    """Atomic ``.tmp → rename`` write of ``schema_report.json``.

    ``ensure_ascii=False`` so the Cyrillic ``TARGET_COL`` survives
    unescaped (Requirement 4.7). ``default=str`` so pandas ``Timestamp``
    and numpy scalar values inside ``failure_cases`` serialize cleanly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records, truncated = _failure_cases_to_records(failure_cases)
    payload: dict[str, Any] = {
        "schema_version": "0.22.1",
        "timestamp": _now_iso_utc(),
        "source_path": source_path,
        "rows": int(rows),
        "status": status,
        "failure_cases": records,
        "failure_cases_truncated": truncated,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
            default=str,
        ),
        encoding="utf-8",
    )
    tmp_path.replace(path)


# --------------------------------------------------------------------------- #
# Fatal-vs-warning classification
# --------------------------------------------------------------------------- #

def _classify_generation_fatals(
    failure_cases: pd.DataFrame,
) -> list[str]:
    """Return a list of fatal-class descriptions found in the lazy report.

    Non-empty return means ``SchemaError`` must be raised. An empty
    return means every violation is warning-class.
    """
    fatals: list[str] = []
    fc = failure_cases

    # Fatal class 1: ``TARGET_COL`` missing / renamed. Pandera reports this
    # as ``schema_context == "DataFrameSchema"`` with ``check ==
    # "column_in_dataframe"`` and ``failure_case`` equal to the missing
    # column name.
    column_missing = fc[
        (fc["schema_context"] == "DataFrameSchema")
        & (fc["check"] == "column_in_dataframe")
    ]
    if (column_missing["failure_case"] == TARGET_COL).any():
        fatals.append(f"TARGET_COL column missing/renamed ({TARGET_COL!r})")

    # Fatal class 2: duplicate ``TIMESTAMP_COL``. Pandera's ``unique=True``
    # produces ``check == "field_uniqueness"``.
    dup_ts = fc[
        (fc["column"] == TIMESTAMP_COL) & (fc["check"] == "field_uniqueness")
    ]
    if len(dup_ts) > 0:
        fatals.append(
            f"duplicate {TIMESTAMP_COL!r}: {len(dup_ts)} offending row(s)"
        )

    # Fatal class 3: ``TARGET_COL`` value ``> CAPACITY_MW``. Pandera
    # reports the in_range check; we filter to values strictly above the
    # upper bound (below-zero values are warning-class range deviations
    # captured by the same check but with a negative failure_case).
    over_cap = fc[
        (fc["column"] == TARGET_COL)
        & (fc["check"].astype(str).str.startswith("in_range"))
    ].copy()
    if len(over_cap) > 0:
        try:
            over_cap["__num"] = pd.to_numeric(
                over_cap["failure_case"], errors="coerce"
            )
            over_max = over_cap[over_cap["__num"] > CAPACITY_MW]
            if len(over_max) > 0:
                fatals.append(
                    f"{TARGET_COL!r} over-capacity: "
                    f"{len(over_max)} value(s) > {CAPACITY_MW}"
                )
        except Exception:  # pragma: no cover — defensive; numeric coercion
            # Err on the side of reporting fatal so a weird non-numeric
            # target never slips through.
            fatals.append(
                f"{TARGET_COL!r} range violation (non-numeric failure_case)"
            )

    return fatals


def _emit_range_warnings(
    failure_cases: pd.DataFrame,
    *,
    suppress_columns: tuple[str, ...] = (),
) -> None:
    """Emit one ``DataQualityWarning`` per ``(column, check)`` class.

    Per Requirement 4.6 the schema validator logs one warning per
    violation class, not one per row. ``suppress_columns`` lets callers
    skip columns already reported as fatal (e.g. ``TARGET_COL`` when a
    warn-class row also exists for it) so the exception message and the
    warnings don't double up.
    """
    fc = failure_cases
    if len(fc) == 0:
        return
    # One row per distinct (column, check) pair, counting occurrences.
    grouped = (
        fc.assign(
            column=fc["column"].fillna("<dataframe>"),
            check=fc["check"].fillna("<unknown>"),
        )
        .groupby(["column", "check"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    for _, row in grouped.iterrows():
        column = row["column"]
        if column in suppress_columns:
            continue
        check = row["check"]
        count = int(row["count"])
        warnings.warn(
            f"schema range violation: column={column!r} check={check!r} "
            f"count={count}",
            DataQualityWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------- #
# validate_* public API
# --------------------------------------------------------------------------- #

def _resolve_source_path(source_path: str | Path | None) -> str:
    if source_path is None:
        return "n/a"
    return str(source_path)


def _run_lazy(
    schema: pa.DataFrameSchema,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run ``schema.validate(df, lazy=True)`` and return the validated
    frame plus the accumulated ``failure_cases`` frame (or ``None`` on
    success).
    """
    try:
        validated = schema.validate(df, lazy=True)
        return validated, None
    except pa_errors.SchemaErrors as err:
        return df, err.failure_cases


def validate_generation(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate a training-generation frame against :data:`GenerationSchema`.

    Writes ``data/interim/schema_report.json`` atomically and returns
    the validated DataFrame. Raises :class:`SchemaError` on any fatal
    violation class (duplicate timestamp, over-capacity target, or
    ``TARGET_COL`` rename).
    """
    report = Path(report_path) if report_path is not None else SCHEMA_REPORT_PATH
    src = _resolve_source_path(source_path)
    validated, failure_cases = _run_lazy(GenerationSchema, df)

    if failure_cases is None:
        _write_report_atomic(
            path=report,
            source_path=src,
            rows=len(df),
            status="ok",
            failure_cases=None,
        )
        return validated

    fatals = _classify_generation_fatals(failure_cases)
    status = "warn"  # "fatal" is never persisted — we raise instead.

    # Write the report first so the jury note can cite it even when the
    # loader aborts on a fatal class.
    _write_report_atomic(
        path=report,
        source_path=src,
        rows=len(df),
        status=status,
        failure_cases=failure_cases,
    )

    if fatals:
        raise SchemaError(
            "GenerationSchema fatal violation(s): " + "; ".join(fatals)
        )

    # Warning-class only. Emit one warning per (column, check) class.
    _emit_range_warnings(failure_cases)
    return validated


def validate_weather(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate a weather-feature frame against :data:`WeatherSchema`.

    Weather violations are warning-class by contract — there is no
    target column here, so nothing can be fatal. Emits one
    ``DataQualityWarning`` per violation class, writes the report, and
    returns the validated frame.

    TODO (Task 6.1): the active-turbine correlation for
    ``wind_speed_* > 35 m/s ∧ Active_Turbine_Count == 0`` cannot be
    evaluated here because the turbine count is not a WeatherSchema
    column. Loader wiring (Task 6.1) will join the two frames and raise
    the secondary soft check at that boundary.
    """
    return _validate_non_fatal(
        WeatherSchema,
        df,
        source_path=source_path,
        report_path=report_path,
        schema_name="WeatherSchema",
    )


def validate_forecast(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate a forecast-feature frame against :data:`ForecastSchema`.

    Same warning-only contract as :func:`validate_weather`.
    """
    return _validate_non_fatal(
        ForecastSchema,
        df,
        source_path=source_path,
        report_path=report_path,
        schema_name="ForecastSchema",
    )


def validate_submission(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate a submission frame against :data:`SubmissionSchema`.

    Any SubmissionSchema violation is **fatal** — a malformed
    submission must never reach the leaderboard. The report is written
    first so the ``.tmp`` left behind by ``write_submission`` can be
    diagnosed against it.
    """
    report = Path(report_path) if report_path is not None else SCHEMA_REPORT_PATH
    src = _resolve_source_path(source_path)
    validated, failure_cases = _run_lazy(SubmissionSchema, df)

    if failure_cases is None:
        _write_report_atomic(
            path=report,
            source_path=src,
            rows=len(df),
            status="ok",
            failure_cases=None,
        )
        return validated

    _write_report_atomic(
        path=report,
        source_path=src,
        rows=len(df),
        status="warn",
        failure_cases=failure_cases,
    )
    # Summarise offending classes in the message for quick triage.
    summary = (
        failure_cases.assign(
            column=failure_cases["column"].fillna("<dataframe>"),
            check=failure_cases["check"].fillna("<unknown>"),
        )
        .groupby(["column", "check"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    classes = "; ".join(
        f"{row['column']!r}/{row['check']!r}×{int(row['count'])}"
        for _, row in summary.iterrows()
    )
    raise SchemaError(f"SubmissionSchema violation(s): {classes}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _validate_non_fatal(
    schema: pa.DataFrameSchema,
    df: pd.DataFrame,
    *,
    source_path: str | Path | None,
    report_path: str | Path | None,
    schema_name: str,
) -> pd.DataFrame:
    """Shared body for the Weather / Forecast validators.

    Both schemas treat every collected violation as warning-class.
    """
    report = Path(report_path) if report_path is not None else SCHEMA_REPORT_PATH
    src = _resolve_source_path(source_path)
    validated, failure_cases = _run_lazy(schema, df)

    if failure_cases is None:
        _write_report_atomic(
            path=report,
            source_path=src,
            rows=len(df),
            status="ok",
            failure_cases=None,
        )
        return validated

    _write_report_atomic(
        path=report,
        source_path=src,
        rows=len(df),
        status="warn",
        failure_cases=failure_cases,
    )
    _emit_range_warnings(failure_cases)
    # The schema name is recorded implicitly via the per-class warning
    # messages; we don't need to surface it separately for warning-only
    # schemas. ``schema_name`` is kept as a parameter for future logging.
    del schema_name
    return validated
