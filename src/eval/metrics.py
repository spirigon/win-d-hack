"""Official competition metric and helpers.

The organizer's score (from the hackathon brief) is the hourly normalized MAE
expressed as a percentage of installed capacity:

    nMAE (%) = mean(|y_true - y_pred|) / N_ust * 100

where ``N_ust = 90.09 MW`` is the installed capacity of the farm.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.data.schema import CAPACITY_MW


def normalized_mae(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    capacity: float = CAPACITY_MW,
) -> float:
    """Return nMAE as a percentage of installed capacity.

    Predictions are clipped to ``[0, capacity]`` internally because the
    evaluator clips too; rewarding over-capacity predictions would otherwise
    double-penalize.
    """
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true_arr.shape} y_pred={y_pred_arr.shape}"
        )
    y_pred_clipped = np.clip(y_pred_arr, 0.0, capacity)
    return float(np.mean(np.abs(y_true_arr - y_pred_clipped)) / capacity * 100.0)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))
