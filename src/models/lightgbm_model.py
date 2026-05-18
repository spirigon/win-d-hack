"""LightGBM regressor wrapper for wind power forecasting.

Design:
- Single global model (no per-horizon split) because the task is a flat
  per-row regression: given weather features at time t, predict power at t.
- MAE (L1) objective aligns directly with the competition metric (nMAE).
- ``deterministic=True, force_col_wise=True`` for reproducibility.
- Early stopping on a held-out validation set during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np


@dataclass
class LGBMConfig:
    """Hyperparameters for the LightGBM regressor."""

    objective: str = "regression_l1"  # MAE
    metric: str = "mae"
    num_leaves: int = 127
    learning_rate: float = 0.05
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    min_data_in_leaf: int = 20
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    num_boost_round: int = 4000
    early_stopping_rounds: int = 200
    verbose: int = -1
    seed: int = 42
    deterministic: bool = True
    force_col_wise: bool = True
    n_jobs: int = -1
    log_period: int = 200  # set 0 to silence per-fit logs (Optuna)

    def to_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "verbose": self.verbose,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "force_col_wise": self.force_col_wise,
            "n_jobs": self.n_jobs,
        }


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    config: LGBMConfig | None = None,
) -> lgb.Booster:
    """Train a LightGBM model and return the booster.

    If ``X_val`` / ``y_val`` are provided, early stopping is used.
    """
    if config is None:
        config = LGBMConfig()

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)

    callbacks: list[Any] = []
    if config.log_period > 0:
        callbacks.append(lgb.log_evaluation(period=config.log_period))
    valid_sets = [dtrain]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, free_raw_data=False)
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(
            lgb.early_stopping(config.early_stopping_rounds, verbose=config.log_period > 0)
        )

    booster = lgb.train(
        config.to_params(),
        dtrain,
        num_boost_round=config.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def predict_lgbm(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predict and clip to [0, 90.09]."""
    preds = booster.predict(X, num_iteration=booster.best_iteration)
    return np.clip(preds, 0.0, 90.09)
