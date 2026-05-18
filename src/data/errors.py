"""Shared exception types for the data-quality hardening pack.

A standalone module holding the error classes that are raised by more than
one hardening component. Keeping them here avoids circular imports between
`src/data/outliers.py`, `src/data/residual_cleaner.py`, and
`src/inference/submission.py`: each module imports from
`src.data.errors` rather than from each other.

Note
----
`SchemaError` is intentionally *not* declared here; it lives next to the
pandera schemas in `src/data/schemas.py` because it is only ever raised by
the Schema_Validator. Only errors shared across modules belong in this file.
"""

from __future__ import annotations


class CleanerSanityError(Exception):
    """Raised when a Cleaner rule flags more of the training set than the
    design contract allows.

    Two callers raise this:

    - `src.data.outliers`: fires when any impossible-row rule flags more
      than 5 % of the training set (Requirement 2.8) — the rule itself is
      broken, not the data.
    - `src.data.residual_cleaner`: fires when the residual cleaner would
      flag more than 2 % of the training set (Requirement 6.5) — same
      semantics applied to label-noise candidates.

    The message SHOULD name the offending rule and the observed fraction so
    diagnosis does not require re-running the pipeline.
    """


class SanityGateError(Exception):
    """Raised by the Sanity_Gate_Suite inside `src.inference.submission`
    when any enforced submission gate fails (Requirement 5.7).

    The message SHOULD name the failing gate (e.g. ``"range"``,
    ``"row_count"``, ``"duplicate_timestamp"``, ``"wind_sanity"``,
    ``"pct95_capacity"``, ``"wind_monotonicity"``) and the offending rows
    so the `.tmp` file left on disk can be inspected directly.
    """


__all__ = ["CleanerSanityError", "SanityGateError"]
