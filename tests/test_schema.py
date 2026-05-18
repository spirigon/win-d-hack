"""Smoke tests for src.data.schemas — Task 5 checkpoint stubs.

These are minimal passing examples that confirm the module imports and the
basic public API works. Full property-based tests live in the optional
sub-tasks (2.2–2.6) and are not dispatched yet.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TURBINES_IN_MAINTENANCE_COL
from src.data.schemas import (
    GenerationSchema,
    WeatherSchema,
    ForecastSchema,
    SubmissionSchema,
    SchemaError,
    validate_generation,
    validate_weather,
    validate_forecast,
    validate_submission,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generation_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal valid generation frame."""
    return pd.DataFrame(
        {
            TIMESTAMP_COL: pd.date_range("2023-01-01", periods=n, freq="h"),
            TARGET_COL: [10.0, 20.0, 30.0][:n],
            TURBINES_IN_MAINTENANCE_COL: [0, 1, 2][:n],
        }
    )


def _make_weather_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal valid weather frame."""
    return pd.DataFrame(
        {
            "wind_speed_10m": [5.0, 6.0, 7.0][:n],
            "wind_speed_80m": [8.0, 9.0, 10.0][:n],
            "wind_speed_120m": [9.0, 10.0, 11.0][:n],
            "wind_speed_180m": [10.0, 11.0, 12.0][:n],
            "wind_gusts_10m": [12.0, 13.0, 14.0][:n],
            "temperature_2m": [15.0, 16.0, 17.0][:n],
        }
    )


def _make_submission_df(n: int = 24) -> pd.DataFrame:
    """Build a minimal valid submission frame (24-row Day-i variant)."""
    return pd.DataFrame(
        {
            TIMESTAMP_COL: pd.date_range("2026-05-18", periods=n, freq="h"),
            TARGET_COL: [float(i) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# GenerationSchema smoke test
# ---------------------------------------------------------------------------

def test_validate_generation_valid_frame(tmp_path: Path) -> None:
    """validate_generation returns the frame unchanged on a valid 3-row input."""
    df = _make_generation_df(3)
    report = tmp_path / "schema_report.json"
    result = validate_generation(df, report_path=report)
    assert len(result) == 3
    assert list(result.columns) == list(df.columns)
    assert report.exists(), "schema_report.json should be written"


def test_validate_generation_fatal_over_capacity(tmp_path: Path) -> None:
    """validate_generation raises SchemaError when TARGET_COL > CAPACITY_MW."""
    df = _make_generation_df(3)
    df.loc[0, TARGET_COL] = CAPACITY_MW + 1.0
    report = tmp_path / "schema_report.json"
    with pytest.raises(SchemaError, match="over-capacity"):
        validate_generation(df, report_path=report)


def test_validate_generation_fatal_duplicate_timestamp(tmp_path: Path) -> None:
    """validate_generation raises SchemaError on duplicate timestamps."""
    df = _make_generation_df(3)
    df.loc[1, TIMESTAMP_COL] = df.loc[0, TIMESTAMP_COL]  # duplicate
    report = tmp_path / "schema_report.json"
    with pytest.raises(SchemaError, match="duplicate"):
        validate_generation(df, report_path=report)


# ---------------------------------------------------------------------------
# WeatherSchema smoke test
# ---------------------------------------------------------------------------

def test_validate_weather_valid_frame(tmp_path: Path) -> None:
    """validate_weather returns the frame unchanged on a valid 3-row input."""
    df = _make_weather_df(3)
    report = tmp_path / "schema_report.json"
    result = validate_weather(df, report_path=report)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# ForecastSchema smoke test
# ---------------------------------------------------------------------------

def test_validate_forecast_valid_frame(tmp_path: Path) -> None:
    """validate_forecast accepts a valid weather-like frame (no lead_hour)."""
    df = _make_weather_df(3)
    report = tmp_path / "schema_report.json"
    result = validate_forecast(df, report_path=report)
    assert len(result) == 3


def test_validate_forecast_with_lead_hour(tmp_path: Path) -> None:
    """validate_forecast accepts a frame that includes the optional lead_hour column."""
    df = _make_weather_df(3)
    df["lead_hour"] = [0, 1, 2]
    report = tmp_path / "schema_report.json"
    result = validate_forecast(df, report_path=report)
    assert "lead_hour" in result.columns


# ---------------------------------------------------------------------------
# SubmissionSchema smoke test
# ---------------------------------------------------------------------------

def test_validate_submission_valid_24_rows(tmp_path: Path) -> None:
    """validate_submission accepts a valid 24-row Day-i submission."""
    df = _make_submission_df(24)
    report = tmp_path / "schema_report.json"
    result = validate_submission(df, report_path=report)
    assert len(result) == 24


def test_validate_submission_wrong_row_count_raises(tmp_path: Path) -> None:
    """validate_submission raises SchemaError when row count is not in {2126, 24}."""
    df = _make_submission_df(10)  # 10 rows — invalid
    report = tmp_path / "schema_report.json"
    with pytest.raises(SchemaError):
        validate_submission(df, report_path=report)
