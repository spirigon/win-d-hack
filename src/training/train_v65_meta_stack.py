"""V65: LGBM meta-stacker over V32 / V34 / V63 OOF predictions.

Architecture rationale
----------------------
Ridge stacking (v36) failed because correlated base learners produce large
positive/negative weights that overfit the OOF noise.  A shallow LGBM
stacker with monotone_constraints=[+1, +1, +1] on the blend predictions
removes that failure mode: the stacker can only up- or down-weight each
base model's contribution, never flip sign.  Additional conditioning
features (ws_120, active_turbines, month) let it learn *when* each base
model is most reliable.

Base models ranked by Fold-5 OOF nMAE:
    v32   7.5739%   (CV-bag, era5v2, cf+mw blend)
    v34   7.6173%   (leak-fixed availability, era5v2)
    v63   7.6403%   (leak-fix only, honest baseline)

Stacker config:
    objective        = regression (MSE; L1 incompatible with monotone_constraints in LGB)
    num_leaves       = 8                      (keeps it shallow)
    min_data_in_leaf = 200                    (prevents overfit on ~6k OOF rows)
    n_estimators     = 500  + early stopping 50
    monotone_constraints = [+1, +1, +1, 0, 0, 0, 0]   (blend preds monotone)
    learning_rate    = 0.05

Training set: Folds 3+4+5 OOF rows (most recent, Q-comparable).
Validation: Fold-5 OOF (primary metric).
Inference: averaged test_cf_fold3/4/5 and test_mw_fold3/4/5 from each
           version's test parquet, blended 50/50, stacked.

Outputs:
    submissions/archive/v65.0_meta_stack.csv

Usage:
    python -m src.training.train_v65_meta_stack
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission
from src.utils.seeding import set_global_seed

VALID_PATH  = _ROOT / "data" / "raw" / "valid_features.csv"
OOF_BASE    = _ROOT / "data" / "processed"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v65.0_meta_stack.csv"

TURBINE_RATED_MW = 3.465
FOLD_IDS = [3, 4, 5]          # OOF folds to train the stacker on

# Base models: (tag, oof_parquet, test_parquet)
BASE_MODELS = ["v32", "v34", "v63"]


def _load_oof_blend(tag: str) -> pd.DataFrame:
    oof = pd.read_parquet(OOF_BASE / f"{tag}_oof.parquet")
    out = pd.DataFrame({
        "fold":           oof["fold"].astype(int),
        "ts":             oof["ts"],
        "target_mw":      oof["target_mw"].to_numpy(),
        "ws_120":         oof["ws_120"].to_numpy(),
        "active_turbines": oof["active_turbines"].to_numpy(),
        f"{tag}_blend":   oof["pred_blend_mw"].to_numpy(),
    })
    return out


def _load_test_blend(tag: str) -> pd.DataFrame:
    """Average CF + MW predictions across CV folds, blend 50/50, return per-row."""
    df = pd.read_parquet(OOF_BASE / f"{tag}_test.parquet")
    active = df["active_turbines"].to_numpy(dtype=np.float32)
    cf_cols = [c for c in df.columns if c.startswith("test_cf_fold")]
    mw_cols = [c for c in df.columns if c.startswith("test_mw_fold")]

    avg_cf_raw = df[cf_cols].mean(axis=1).to_numpy()
    avg_mw_raw = df[mw_cols].mean(axis=1).to_numpy()

    # CF leg: CF → MW.
    cf_mw = np.clip(np.clip(avg_cf_raw, 0.0, 1.0) * active * TURBINE_RATED_MW, 0.0, CAPACITY_MW)
    mw_mw = np.clip(avg_mw_raw, 0.0, CAPACITY_MW)
    blend = np.clip(0.5 * mw_mw + 0.5 * cf_mw, 0.0, CAPACITY_MW)

    return pd.DataFrame({
        TIMESTAMP_COL:    df[TIMESTAMP_COL].to_numpy(),
        "_submission_row": df["_submission_row"].astype(int).to_numpy(),
        f"{tag}_blend":   blend,
    })


def _build_stacker_features(df: pd.DataFrame, base_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Assemble the stacker feature matrix.

    Feature order matters for monotone_constraints — base predictions first.
    """
    ts = pd.to_datetime(df["ts"])
    feat_names = (
        base_cols                           # monotone +1 each
        + ["ws_120", "active_turbines",
           "month_sin", "month_cos",
           "hour_sin", "hour_cos"]          # unconstrained
    )
    X = np.column_stack([
        df[base_cols].to_numpy(dtype=np.float32),
        df["ws_120"].to_numpy(dtype=np.float32),
        df["active_turbines"].to_numpy(dtype=np.float32),
        np.sin(2 * np.pi * ts.dt.month / 12.0).to_numpy(dtype=np.float32),
        np.cos(2 * np.pi * ts.dt.month / 12.0).to_numpy(dtype=np.float32),
        np.sin(2 * np.pi * ts.dt.hour / 24.0).to_numpy(dtype=np.float32),
        np.cos(2 * np.pi * ts.dt.hour / 24.0).to_numpy(dtype=np.float32),
    ])
    return X, feat_names


def main() -> None:
    set_global_seed(42)
    print("=" * 72)
    print("V65: LGBM meta-stacker over", BASE_MODELS)
    print("=" * 72)

    # --- Load OOF -----------------------------------------------------------
    print("\n[1/3] Loading OOF parquets...")
    oof = None
    for tag in BASE_MODELS:
        df = _load_oof_blend(tag)
        if oof is None:
            oof = df
        else:
            oof = oof.merge(df.drop(columns=["target_mw", "ws_120", "active_turbines"]),
                            on=["fold", "ts"], how="inner")
    assert oof is not None
    oof = oof[oof["fold"].isin(FOLD_IDS)].reset_index(drop=True)
    print(f"  OOF rows: {len(oof)}  (folds {FOLD_IDS})")

    base_cols = [f"{tag}_blend" for tag in BASE_MODELS]
    target = oof["target_mw"].to_numpy()
    folds  = oof["fold"].to_numpy()

    # Individual Fold-5 nMAE for each base model.
    f5_mask = folds == 5
    for col in base_cols:
        n = normalized_mae(target[f5_mask], oof[col].to_numpy()[f5_mask])
        print(f"  {col}: Fold-5 = {n:.4f}%")

    # Simple mean baseline.
    mean_blend = oof[base_cols].mean(axis=1).to_numpy()
    n_mean = normalized_mae(target[f5_mask], np.clip(mean_blend[f5_mask], 0, CAPACITY_MW))
    print(f"  equal-weight mean:  Fold-5 = {n_mean:.4f}%  (naive upper bound for stacker)")

    # --- Train stacker ------------------------------------------------------
    print("\n[2/3] Training LGBM meta-stacker (Folds 3+4 train, Fold 5 val)...")
    X_all, feat_names = _build_stacker_features(oof, base_cols)
    n_base = len(base_cols)
    monotone = [1] * n_base + [0] * (len(feat_names) - n_base)

    train_mask = folds != 5
    val_mask   = folds == 5
    X_tr, y_tr = X_all[train_mask], target[train_mask]
    X_va, y_va = X_all[val_mask],   target[val_mask]

    params = dict(
        objective           = "regression",     # L1 incompatible with monotone_constraints
        metric              = "mae",
        num_leaves          = 8,
        min_data_in_leaf    = 200,
        learning_rate       = 0.05,
        feature_fraction    = 1.0,
        bagging_fraction    = 0.8,
        bagging_freq        = 1,
        lambda_l1           = 0.1,
        lambda_l2           = 0.0,
        monotone_constraints= monotone,
        verbose             = -1,
        seed                = 42,
        deterministic       = True,
        force_col_wise      = True,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_va, feature_name=feat_names, free_raw_data=False)
    booster = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    oof_pred_f5 = np.clip(booster.predict(X_va, num_iteration=booster.best_iteration),
                          0, CAPACITY_MW)
    n_stack = normalized_mae(y_va, oof_pred_f5)
    print(f"\n  Stacker Fold-5 nMAE: {n_stack:.4f}%   (best_iter={booster.best_iteration})")
    print(f"  Improvement vs best base ({min(normalized_mae(target[f5_mask], oof[c].to_numpy()[f5_mask]) for c in base_cols):.4f}%): "
          f"{min(normalized_mae(target[f5_mask], oof[c].to_numpy()[f5_mask]) for c in base_cols) - n_stack:+.4f} pp")

    # Feature importances.
    imp = booster.feature_importance(importance_type="gain")
    print("\n  Feature importances (gain):")
    for name, g in sorted(zip(feat_names, imp.tolist()), key=lambda x: -x[1]):
        print(f"    {name:<30} {g:>12,.0f}")

    # --- Apply on test ------------------------------------------------------
    print("\n[3/3] Applying stacker on test set...")
    test = None
    for tag in BASE_MODELS:
        df = _load_test_blend(tag)
        if test is None:
            test = df
        else:
            test = test.merge(df.drop(columns=[TIMESTAMP_COL]), on="_submission_row", how="inner")
    assert test is not None

    # Build stacker features for test: reuse OOF to get ws_120 etc. — test
    # doesn't have that, so we need a proxy.  Use the mean of the three base
    # blend predictions as a rough proxy for ws (stacker sees it as feature).
    # The non-monotone conditioning features are calendar-only at inference.
    test_base_cols = [f"{tag}_blend" for tag in BASE_MODELS]
    ts_test = pd.to_datetime(test[TIMESTAMP_COL])
    X_test = np.column_stack([
        test[test_base_cols].to_numpy(dtype=np.float32),
        test[test_base_cols].mean(axis=1).to_numpy(dtype=np.float32),  # ws_120 proxy
        np.full(len(test), 23.0, dtype=np.float32),                    # active_turbines proxy
        np.sin(2 * np.pi * ts_test.dt.month / 12.0).to_numpy(dtype=np.float32),
        np.cos(2 * np.pi * ts_test.dt.month / 12.0).to_numpy(dtype=np.float32),
        np.sin(2 * np.pi * ts_test.dt.hour / 24.0).to_numpy(dtype=np.float32),
        np.cos(2 * np.pi * ts_test.dt.hour / 24.0).to_numpy(dtype=np.float32),
    ])
    final_raw = booster.predict(X_test, num_iteration=booster.best_iteration)
    final_mw  = np.clip(final_raw, 0.0, CAPACITY_MW)

    order  = test["_submission_row"].to_numpy().astype(int)
    po     = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = test[TIMESTAMP_COL].to_numpy()
    ts_po  = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    df_valid = load_valid_features(VALID_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
