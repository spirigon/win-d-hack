"""Walk-forward expanding-window folds for time-series CV.

Fold-5 ends at 2025-12-31 with validation window 2025-01-01 .. 2025-03-31,
i.e. Q1 of the most recent full year. This is our direct surrogate for the
hold-out metric on Q1 2026.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import TIMESTAMP_COL


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp  # inclusive
    val_start: pd.Timestamp  # inclusive
    val_end: pd.Timestamp  # inclusive


def default_folds() -> list[Fold]:
    """Walk-forward expanding folds, last one dedicated to Q1-of-final-year.

    Each validation window is a full calendar quarter so per-fold nMAE numbers
    are comparable across folds.
    """
    ts = pd.Timestamp
    return [
        Fold("fold1_2023Q1", ts("2022-12-31 23:00:00"), ts("2023-01-01"), ts("2023-03-31 23:00:00")),
        Fold("fold2_2023Q4", ts("2023-09-30 23:00:00"), ts("2023-10-01"), ts("2023-12-31 23:00:00")),
        Fold("fold3_2024Q2", ts("2024-03-31 23:00:00"), ts("2024-04-01"), ts("2024-06-30 23:00:00")),
        Fold("fold4_2024Q4", ts("2024-09-30 23:00:00"), ts("2024-10-01"), ts("2024-12-31 23:00:00")),
        # Final fold is the Q1-2026 surrogate (same calendar quarter as the target).
        Fold("fold5_2025Q1", ts("2024-12-31 23:00:00"), ts("2025-01-01"), ts("2025-03-31 23:00:00")),
    ]


def split_indices(df: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) masks for a fold against ``df``."""
    if TIMESTAMP_COL not in df.columns:
        raise KeyError(f"DataFrame must contain {TIMESTAMP_COL!r}")
    ts = df[TIMESTAMP_COL]
    train_mask = ts <= fold.train_end
    val_mask = (ts >= fold.val_start) & (ts <= fold.val_end)
    if not (train_mask & val_mask).sum() == 0:
        raise AssertionError(f"Fold {fold.name} has train/val overlap.")
    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    return train_idx, val_idx
