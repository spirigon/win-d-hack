"""Pytest configuration for the wind-d test suite.

Runs at session start (via ``pytest_configure``):

* Seeds Python / NumPy / PyTorch deterministically via
  ``src.utils.seeding.set_global_seed(42)``. Every test module therefore
  starts from the same RNG state, matching the production determinism
  contract in ``ARCHITECTURE.md`` §10.
* Registers a targeted ``filterwarnings`` entry that allows
  ``src.utils.warnings.DataQualityWarning`` to be emitted as a *warning*
  instead of being promoted to an error. The global
  ``filterwarnings = ["error", ...]`` policy in ``pyproject.toml`` stays in
  place for every other warning class, so accidental ``DeprecationWarning``
  / ``RuntimeWarning`` / ``FutureWarning`` regressions still fail tests.

Keeping this logic inside ``conftest.py`` — rather than widening the
pyproject.toml filter — scopes the exception to the test session only and
preserves warnings-as-errors on real pipeline runs.
"""

from __future__ import annotations

from src.utils.seeding import set_global_seed


def pytest_configure(config):
    """Configure the test session: seed RNGs and allow ``DataQualityWarning``."""
    set_global_seed(42)
    # Targeted escape hatch for the project's own data-quality warning class.
    # Format: "action:message:category:module:lineno" — we only pin the category.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::src.utils.warnings.DataQualityWarning",
    )
