"""Smoke tests for src.data.outliers — Task 5 checkpoint stubs.

These are minimal passing examples that confirm the module imports and the
basic public API works. Full property-based tests live in the optional
sub-tasks (3.2–3.9) and are not dispatched yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    CAPACITY_MW,
    TARGET_COL,
    TIMESTAMP_COL,
    TURBINES_IN_MAINTENANCE_COL,
)
from src.data.outliers import (
    IMPOSSIBLE_REASONS,
    identify_impossible_rows,
    impossible_reasons,
    compute_training_weights,
    write_outlier_audit,
    add_imputation_flag,
    add_nwp_era5_disagreement_flag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_df(n: int = 5) -> pd.DataFrame:
    """Build a small frame with no impossible rows."""
    return pd.DataFrame(
        {
            TIMESTAMP_COL: pd.date_range("2023-01-01", periods=n, freq="h"),
            TARGET_COL: [10.0] * n,
            TURBINES_IN_MAINTENANCE_COL: [0] * n,
            "wind_speed_10m": [8.0] * n,
            "wind_speed_80m": [10.0] * n,
            "wind_speed_120m": [11.0] * n,
            "wind_speed_180m": [12.0] * n,
            "wind_gusts_10m": [15.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# identify_impossible_rows smoke tests
# ---------------------------------------------------------------------------

def test_identify_impossible_rows_clean_frame() -> None:
    """identify_impossible_rows returns all-False on a clean 5-row frame."""
    df = _make_clean_df(5)
    mask = identify_impossible_rows(df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert len(mask) == 5
    assert not mask.any(), "Expected no impossible rows in a clean frame"


def test_identify_impossible_rows_over_capacity() -> None:
    """identify_impossible_rows flags a row with target > CAPACITY_MW."""
    df = _make_clean_df(5)
    df.loc[2, TARGET_COL] = CAPACITY_MW + 1.0
    mask = identify_impossible_rows(df)
    assert mask[2], "Row with over-capacity target should be flagged"
    assert mask.sum() == 1


def test_identify_impossible_rows_maintenance_zero() -> None:
    """identify_impossible_rows flags maintenance_zero: 24+ turbines down, target < 1."""
    df = _make_clean_df(5)
    # 24 turbines in maintenance → active = 2, target < 1.0
    df.loc[1, TURBINES_IN_MAINTENANCE_COL] = 24
    df.loc[1, TARGET_COL] = 0.5
    mask = identify_impossible_rows(df)
    assert mask[1], "maintenance_zero row should be flagged"


# ---------------------------------------------------------------------------
# impossible_reasons smoke tests
# ---------------------------------------------------------------------------

def test_impossible_reasons_values_in_enum() -> None:
    """impossible_reasons returns only values from IMPOSSIBLE_REASONS."""
    df = _make_clean_df(5)
    reasons = impossible_reasons(df)
    assert isinstance(reasons, pd.Series)
    for val in reasons:
        assert val in IMPOSSIBLE_REASONS, f"Unexpected reason: {val!r}"


def test_impossible_reasons_priority_over_capacity_wins() -> None:
    """over_capacity takes priority over low_wind_high_power when both match."""
    df = _make_clean_df(5)
    # Trigger over_capacity AND low_wind_high_power on the same row.
    df.loc[0, TARGET_COL] = CAPACITY_MW + 5.0
    df.loc[0, "wind_speed_10m"] = 1.0
    df.loc[0, "wind_speed_80m"] = 1.0
    df.loc[0, "wind_speed_120m"] = 1.0
    df.loc[0, "wind_gusts_10m"] = 1.0
    reasons = impossible_reasons(df)
    assert reasons[0] == "over_capacity"


# ---------------------------------------------------------------------------
# compute_training_weights smoke tests
# ---------------------------------------------------------------------------

def test_compute_training_weights_clean_frame() -> None:
    """compute_training_weights returns all-ones for a clean frame."""
    df = _make_clean_df(5)
    weights = compute_training_weights(df)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == 5
    assert np.allclose(weights, 1.0)


def test_compute_training_weights_impossible_rows_get_zero() -> None:
    """compute_training_weights assigns 0.0 to impossible rows."""
    df = _make_clean_df(5)
    df.loc[3, TARGET_COL] = CAPACITY_MW + 1.0
    weights = compute_training_weights(df)
    assert weights[3] == 0.0
    assert np.allclose(weights[[0, 1, 2, 4]], 1.0)


# ---------------------------------------------------------------------------
# write_outlier_audit smoke test
# ---------------------------------------------------------------------------

def test_write_outlier_audit_creates_file(tmp_path: Path) -> None:
    """write_outlier_audit creates a CSV file with the expected header comment."""
    import warnings
    df = _make_clean_df(5)
    df.loc[0, TARGET_COL] = CAPACITY_MW + 1.0  # one impossible row
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        weights = compute_training_weights(df)
    reasons = impossible_reasons(df)
    out = write_outlier_audit(df, weights, reasons, path=tmp_path / "audit.csv")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.startswith("# downweight_2022="), "Header comment missing"
    assert "total_excluded=1" in content


# ---------------------------------------------------------------------------
# add_nwp_era5_disagreement_flag smoke test
# ---------------------------------------------------------------------------

def test_add_nwp_era5_disagreement_flag_no_era5_col() -> None:
    """add_nwp_era5_disagreement_flag returns all-False when ERA5 col is absent."""
    df = _make_clean_df(5)
    result = add_nwp_era5_disagreement_flag(df)
    assert "nwp_era5_disagreement" in result.columns
    assert not result["nwp_era5_disagreement"].any()


def test_add_nwp_era5_disagreement_flag_detects_disagreement() -> None:
    """add_nwp_era5_disagreement_flag flags rows where |ws_120m - era5| > threshold."""
    df = _make_clean_df(5)
    df["era5_wind_speed_100m"] = [11.0, 11.0, 11.0, 11.0, 11.0]
    # Row 0: ws_120m=11.0, era5=11.0 → diff=0 → no flag
    # Row 1: ws_120m=11.0, era5=4.0 → diff=7 > 5 → flag
    df.loc[1, "era5_wind_speed_100m"] = 4.0
    result = add_nwp_era5_disagreement_flag(df, threshold_m_s=5.0)
    assert result.loc[1, "nwp_era5_disagreement"] is True or result.loc[1, "nwp_era5_disagreement"] == True
    assert not result.loc[0, "nwp_era5_disagreement"]
