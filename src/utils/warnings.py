"""Custom warning categories for the pipeline.

`DataQualityWarning` is the single non-fatal channel for data-quality
concerns raised by the Loader, the Cleaner, the Residual_Cleaner, and the
Sanity_Gate_Suite. Using a dedicated `UserWarning` subclass lets callers and
test suites filter on the category without widening the `warnings` filter,
and keeps pytest's `filterwarnings = ["error"]` default honest: only
explicitly allow-listed `DataQualityWarning` instances pass through.

Usage
-----
>>> import warnings
>>> from src.utils.warnings import DataQualityWarning
>>> warnings.warn("180m channel 21.3% imputed", DataQualityWarning, stacklevel=2)
"""

from __future__ import annotations


class DataQualityWarning(UserWarning):
    """Non-fatal data-quality signal.

    Emitted by schema validation (range-class violations), the Cleaner
    (`downweight_2022 < 1.0` diagnostic), and the Sanity_Gate_Suite (skipped
    paired gates when `wind_speed_120m` is absent). Fatal conditions raise
    exceptions from `src.data.errors` instead.
    """


__all__ = ["DataQualityWarning"]
