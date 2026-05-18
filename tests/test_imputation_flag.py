"""Smoke tests for add_imputation_flag — Task 5 checkpoint stubs.

These are minimal passing examples that confirm the module imports and the
basic public API works. Full property-based tests live in the optional
sub-tasks (4.2–4.4) and are not dispatched yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.outliers import add_imputation_flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (df_pre, df_post) pair where rows 1 and 3 are NaN-then-filled."""
    pre_vals = [1.0, np.nan, 3.0, np.nan, 5.0][:n]
    post_vals = [1.0, 2.0, 3.0, 4.0, 5.0][:n]
    df_pre = pd.DataFrame({"wind_speed_180m": pre_vals})
    df_post = pd.DataFrame({"wind_speed_180m": post_vals})
    return df_pre, df_post


# ---------------------------------------------------------------------------
# Basic API smoke tests
# ---------------------------------------------------------------------------

def test_add_imputation_flag_returns_copy() -> None:
    """add_imputation_flag returns a copy of df_post, not the original."""
    df_pre, df_post = _make_pair(5)
    result = add_imputation_flag(df_pre, df_post)
    assert result is not df_post, "Should return a copy, not the original"


def test_add_imputation_flag_column_name() -> None:
    """add_imputation_flag adds 'ws_180m_is_imputed' for the default column."""
    df_pre, df_post = _make_pair(5)
    result = add_imputation_flag(df_pre, df_post)
    assert "ws_180m_is_imputed" in result.columns


def test_add_imputation_flag_correct_values() -> None:
    """add_imputation_flag marks exactly the NaN-then-filled rows as True."""
    df_pre, df_post = _make_pair(5)
    result = add_imputation_flag(df_pre, df_post)
    flag = result["ws_180m_is_imputed"].to_numpy()
    # Rows 1 and 3 were NaN in pre, filled in post → True
    expected = np.array([False, True, False, True, False])
    np.testing.assert_array_equal(flag, expected)


def test_add_imputation_flag_dtype_is_bool() -> None:
    """add_imputation_flag produces a plain bool column (no NaN/NA)."""
    df_pre, df_post = _make_pair(5)
    result = add_imputation_flag(df_pre, df_post)
    assert result["ws_180m_is_imputed"].dtype == np.bool_
    assert not result["ws_180m_is_imputed"].isna().any()


def test_add_imputation_flag_no_nan_in_pre_all_false() -> None:
    """When df_pre has no NaNs, the flag is all-False."""
    df_pre = pd.DataFrame({"wind_speed_180m": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df_post = pd.DataFrame({"wind_speed_180m": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = add_imputation_flag(df_pre, df_post)
    assert not result["ws_180m_is_imputed"].any()


def test_add_imputation_flag_all_nan_in_pre_all_true() -> None:
    """When all df_pre values are NaN and df_post is filled, flag is all-True."""
    df_pre = pd.DataFrame({"wind_speed_180m": [np.nan] * 5})
    df_post = pd.DataFrame({"wind_speed_180m": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = add_imputation_flag(df_pre, df_post)
    assert result["ws_180m_is_imputed"].all()


# ---------------------------------------------------------------------------
# Error-path smoke tests
# ---------------------------------------------------------------------------

def test_add_imputation_flag_length_mismatch_raises() -> None:
    """add_imputation_flag raises ValueError when df_pre and df_post differ in length."""
    df_pre = pd.DataFrame({"wind_speed_180m": [1.0, 2.0, 3.0]})
    df_post = pd.DataFrame({"wind_speed_180m": [1.0, 2.0]})
    with pytest.raises(ValueError, match="length"):
        add_imputation_flag(df_pre, df_post)


def test_add_imputation_flag_missing_col_in_pre_raises() -> None:
    """add_imputation_flag raises KeyError when col is missing from df_pre."""
    df_pre = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
    df_post = pd.DataFrame({"wind_speed_180m": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="df_pre"):
        add_imputation_flag(df_pre, df_post)


def test_add_imputation_flag_missing_col_in_post_raises() -> None:
    """add_imputation_flag raises KeyError when col is missing from df_post."""
    df_pre = pd.DataFrame({"wind_speed_180m": [1.0, 2.0, 3.0]})
    df_post = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="df_post"):
        add_imputation_flag(df_pre, df_post)


def test_add_imputation_flag_custom_col_name() -> None:
    """add_imputation_flag uses f'{col}_is_imputed' for non-default column names."""
    df_pre = pd.DataFrame({"wind_speed_80m": [np.nan, 2.0]})
    df_post = pd.DataFrame({"wind_speed_80m": [1.0, 2.0]})
    result = add_imputation_flag(df_pre, df_post, col="wind_speed_80m")
    assert "wind_speed_80m_is_imputed" in result.columns
    assert result.loc[0, "wind_speed_80m_is_imputed"] is True or result.loc[0, "wind_speed_80m_is_imputed"] == True
    assert not result.loc[1, "wind_speed_80m_is_imputed"]
