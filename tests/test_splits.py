"""Property + unit tests for walk-forward fold construction (Requirement 8.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import TIMESTAMP_COL
from src.data.splits import default_folds, split_indices  # noqa: F401  (split_indices is consumed by Task 10.2)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Inclusive date range covered by the synthetic hourly frame. Spans four
# calendar years (2022-01-01 .. 2025-12-31) so every Fold in
# ``default_folds()`` — including the Fold-5 Q1-2025 window — is exercised
# with at least one hour on both sides of every boundary.
SYNTHETIC_RANGE: tuple[str, str] = ("2022-01-01", "2025-12-31")
HOURS_PER_DAY: int = 24


# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _build_synthetic_frame(rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Return a synthetic hourly frame covering ``SYNTHETIC_RANGE``.

    The frame carries:

    * ``TIMESTAMP_COL`` — hourly UTC timestamps from
      ``2022-01-01 00:00:00`` to ``2025-12-31 23:00:00`` inclusive.
    * ``target`` — random normal values drawn from ``rng`` (or a seeded
      default generator if ``rng`` is ``None``).

    The index is left as a default :class:`pandas.RangeIndex`; the
    timestamp column is **not** promoted to the index because
    :func:`src.data.splits.split_indices` expects ``TIMESTAMP_COL`` to be a
    regular DataFrame column.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    start_str, end_str = SYNTHETIC_RANGE
    timestamps = pd.date_range(
        start=f"{start_str} 00:00:00",
        end=f"{end_str} 23:00:00",
        freq="1h",
        tz="UTC",
    )
    n = len(timestamps)
    return pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            "target": rng.standard_normal(n).astype(np.float64),
        }
    )


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """Session-scoped synthetic hourly frame used by every splits test."""
    return _build_synthetic_frame(np.random.default_rng(42))


# ---------------------------------------------------------------------------
# Scaffolding test
# ---------------------------------------------------------------------------


def test_scaffolding_loads(synthetic_frame: pd.DataFrame) -> None:
    """Sanity-check: the synthetic frame and the splits API are importable."""
    # Expected hourly count derived from the same date_range used by the
    # helper; this captures the 2024 leap day (1461 days × 24 = 35 064 h).
    start_str, end_str = SYNTHETIC_RANGE
    expected_rows = len(
        pd.date_range(
            start=f"{start_str} 00:00:00",
            end=f"{end_str} 23:00:00",
            freq="1h",
            tz="UTC",
        )
    )
    assert len(synthetic_frame) == expected_rows

    folds = list(default_folds())
    assert len(folds) > 0, "default_folds() returned an empty iterable"
