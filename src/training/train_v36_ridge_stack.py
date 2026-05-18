"""V36: Ridge stack over V32 / V34 / V35 OOF predictions.

Reads OOF + per-fold test predictions saved by:

    src/training/train_v32_with_oof.py     → data/processed/v32_oof.parquet, v32_test.parquet
    src/training/train_v34_curtailment.py  → data/processed/v34_oof.parquet, v34_test.parquet
    src/training/train_v35_catboost.py     → data/processed/v35_oof.parquet, v35_test.parquet

Learns ridge stack weights over the leg predictions (CF and MW separately)
using leave-one-fold-out cross-validation so the test estimate is honest:

    Inputs (per row, after iso-recal):
        v32_pred_cf, v32_pred_mw, v34_pred_cf, v34_pred_mw,
        v35_pred_cf, v35_pred_mw,
        wind_speed_120m, hour_of_day_sin, hour_of_day_cos,
        active_turbines, n_repair (if available)

The stack output is the predicted MW directly (no separate CF/MW blend at
the stack level — the ridge learns the optimal mix). Final predictions
are clipped to [0, 90.09].

Outputs:

    submissions/archive/v36.0_ridge_stack.csv

Usage:

    python -m src.training.train_v36_ridge_stack
    python -m src.training.train_v36_ridge_stack --models v32 v34   # subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission
from src.postprocess.iso_recal import apply_iso_recal, fit_iso_recal

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OOF_BASE = _ROOT / "data" / "processed"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v36.0_ridge_stack.csv"

FOLD_IDS = [3, 4, 5]   # the same CV-bag folds


def _load_pair(model_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load OOF and test parquets for a model tag (v32, v34, v35)."""
    oof = pd.read_parquet(OOF_BASE / f"{model_tag}_oof.parquet")
    test = pd.read_parquet(OOF_BASE / f"{model_tag}_test.parquet")
    # Standardise OOF column names so the stacker is uniform across models.
    rename = {
        "pred_cf_mw": f"{model_tag}_pred_cf",
        "pred_mw_mw": f"{model_tag}_pred_mw",
    }
    oof = oof.rename(columns=rename)
    return oof, test


def _build_oof_table(models: list[str]) -> pd.DataFrame:
    """Inner-join OOF dataframes from each model on (fold, ts).

    Returns a single dataframe with shared columns ``fold, ts, target_mw,
    ws_120, active_turbines`` plus per-model ``{tag}_pred_cf``,
    ``{tag}_pred_mw`` columns.
    """
    oofs = {tag: _load_pair(tag)[0] for tag in models}

    base = oofs[models[0]][[
        "fold", "ts", "target_mw", "ws_120", "active_turbines",
        f"{models[0]}_pred_cf", f"{models[0]}_pred_mw",
    ]]
    for tag in models[1:]:
        df = oofs[tag][["fold", "ts", f"{tag}_pred_cf", f"{tag}_pred_mw"]]
        base = base.merge(df, on=["fold", "ts"], how="inner")
    return base


def _build_test_table(models: list[str]) -> pd.DataFrame:
    """Inner-join per-fold test parquets on TIMESTAMP_COL.

    Returns columns:
        TIMESTAMP_COL, _submission_row, active_turbines,
        and ``{tag}_avg_cf``, ``{tag}_avg_mw`` per model — the average
        across the CV-bag folds (3, 4, 5).

    NOTE on units: the per-fold ``test_cf_foldX`` columns saved by the
    train scripts are RAW CF values (range 0-1). We convert each fold's
    CF to MW (via ``active_turbines × 3.465``) BEFORE averaging, so the
    output ``{tag}_avg_cf`` column is already in MW. This way the OOF
    and test CF columns share units (MW) by the time the ridge stacker
    sees them — the OOF parquet's ``pred_cf_mw`` is also already in MW.

    The ``test_mw_foldX`` columns are already in MW; we just average them.
    """
    base = None
    for tag in models:
        df = pd.read_parquet(OOF_BASE / f"{tag}_test.parquet")
        active = df["active_turbines"].to_numpy(dtype=np.float32)
        # CF → MW per fold, then average.
        cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS if f"test_cf_fold{i}" in df.columns]
        cf_mw_per_fold = np.column_stack([
            np.clip(np.clip(df[c].to_numpy(), 0.0, 1.0) * active * 3.465, 0.0, CAPACITY_MW)
            for c in cf_cols
        ])
        df[f"{tag}_avg_cf"] = cf_mw_per_fold.mean(axis=1)

        # MW leg: already in MW.
        mw_cols = [f"test_mw_fold{i}" for i in FOLD_IDS if f"test_mw_fold{i}" in df.columns]
        df[f"{tag}_avg_mw"] = np.clip(df[mw_cols].mean(axis=1), 0.0, CAPACITY_MW)

        keep = [TIMESTAMP_COL, "_submission_row", "active_turbines",
                f"{tag}_avg_cf", f"{tag}_avg_mw"]
        df = df[keep]
        base = df if base is None else base.merge(
            df.drop(columns=["_submission_row", "active_turbines"]),
            on=TIMESTAMP_COL,
            how="inner",
        )
    return base


def _per_leg_iso_recal(
    oof: pd.DataFrame, test: pd.DataFrame, models: list[str], *, enable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply per-leg, per-model iso-recal to OOF and test predictions.

    Both OOF ``{tag}_pred_cf/mw`` columns and test ``{tag}_avg_cf/mw``
    columns are in MW units by this point (see ``_build_test_table``).
    The iso-recal is fit on OOF and applied to both OOF (for diagnostic
    LOO scoring) and the test test predictions.

    If ``enable=False`` the "calibrated" columns simply mirror the raw
    leg predictions (no transformation).

    Returns the OOF and test dataframes with new ``{tag}_cal_cf_mw`` and
    ``{tag}_cal_mw_mw`` columns added.
    """
    oof = oof.copy()
    test = test.copy()

    for tag in models:
        # CF leg.
        oof_cf_mw = np.clip(oof[f"{tag}_pred_cf"].to_numpy(), 0.0, CAPACITY_MW)
        test_cf_mw = np.clip(test[f"{tag}_avg_cf"].to_numpy(), 0.0, CAPACITY_MW)

        if enable:
            ir_cf = fit_iso_recal(oof_cf_mw, oof["target_mw"].to_numpy())
            oof[f"{tag}_cal_cf_mw"] = apply_iso_recal(ir_cf, oof_cf_mw)
            test[f"{tag}_cal_cf_mw"] = apply_iso_recal(ir_cf, test_cf_mw)
        else:
            oof[f"{tag}_cal_cf_mw"] = oof_cf_mw
            test[f"{tag}_cal_cf_mw"] = test_cf_mw

        # MW leg.
        oof_mw_mw = np.clip(oof[f"{tag}_pred_mw"].to_numpy(), 0.0, CAPACITY_MW)
        test_mw_mw = np.clip(test[f"{tag}_avg_mw"].to_numpy(), 0.0, CAPACITY_MW)

        if enable:
            ir_mw = fit_iso_recal(oof_mw_mw, oof["target_mw"].to_numpy())
            oof[f"{tag}_cal_mw_mw"] = apply_iso_recal(ir_mw, oof_mw_mw)
            test[f"{tag}_cal_mw_mw"] = apply_iso_recal(ir_mw, test_mw_mw)
        else:
            oof[f"{tag}_cal_mw_mw"] = oof_mw_mw
            test[f"{tag}_cal_mw_mw"] = test_mw_mw

    return oof, test


def _build_stacker_X(df: pd.DataFrame, models: list[str], *, include_extras: bool) -> np.ndarray:
    """Assemble the ridge feature matrix from calibrated leg predictions."""
    cols = []
    for tag in models:
        cols.append(df[f"{tag}_cal_cf_mw"].to_numpy())
        cols.append(df[f"{tag}_cal_mw_mw"].to_numpy())

    if include_extras:
        ws = df["ws_120"].to_numpy() if "ws_120" in df.columns else df.get(
            "wind_speed_120m", pd.Series(np.zeros(len(df)))
        ).to_numpy()
        cols.append(ws)
        cols.append(ws ** 2)
        # Hour cyclic if available — derived from ts column.
        if "ts" in df.columns:
            ts = pd.to_datetime(df["ts"])
        elif TIMESTAMP_COL in df.columns:
            ts = pd.to_datetime(df[TIMESTAMP_COL])
        else:
            ts = None
        if ts is not None:
            hour = ts.dt.hour
            cols.append(np.sin(2 * np.pi * hour / 24.0).to_numpy())
            cols.append(np.cos(2 * np.pi * hour / 24.0).to_numpy())
        if "active_turbines" in df.columns:
            cols.append(df["active_turbines"].to_numpy())
    return np.column_stack(cols).astype(np.float64)


def _ridge_loo_eval(
    oof: pd.DataFrame, models: list[str], *, alphas: list[float], include_extras: bool,
) -> tuple[float, float]:
    """Leave-one-fold-out ridge evaluation. Returns (best_alpha, mean_nmae_pct)."""
    folds = sorted(oof["fold"].unique())
    best_alpha = None
    best_nmae = np.inf
    for alpha in alphas:
        per_fold = []
        for fold_id in folds:
            train_mask = oof["fold"] != fold_id
            eval_mask = oof["fold"] == fold_id

            X_tr = _build_stacker_X(oof[train_mask], models, include_extras=include_extras)
            y_tr = oof.loc[train_mask, "target_mw"].to_numpy()
            X_va = _build_stacker_X(oof[eval_mask], models, include_extras=include_extras)
            y_va = oof.loc[eval_mask, "target_mw"].to_numpy()

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr)
            preds = np.clip(ridge.predict(X_va_s), 0.0, CAPACITY_MW)
            per_fold.append(float(normalized_mae(y_va, preds)))

        mean_nmae = float(np.mean(per_fold))
        marker = "  ←" if mean_nmae < best_nmae else ""
        print(f"  alpha={alpha:>6.2f}  per-fold nMAE: "
              f"{['%.4f' % v for v in per_fold]}  mean={mean_nmae:.4f}%{marker}")
        if mean_nmae < best_nmae:
            best_nmae = mean_nmae
            best_alpha = alpha
    return best_alpha, best_nmae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v32", "v34", "v35"],
                    help="OOF tags to consume (default: v32 v34 v35)")
    ap.add_argument("--no-iso-recal", action="store_true",
                    help="skip per-leg iso-recal — recommended given v34 LOO findings")
    ap.add_argument("--no-extras", action="store_true",
                    help="disable wind/hour features, use leg preds only")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    args = ap.parse_args()

    print("=" * 72)
    print(f"V36: Ridge stack over {args.models}")
    print("=" * 72)

    # --- Load OOFs -------------------------------------------------------
    print("\n[1/5] Loading OOF and test parquets...")
    for tag in args.models:
        for kind in ("oof", "test"):
            path = OOF_BASE / f"{tag}_{kind}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found — run train_{tag}_*.py first")
    oof = _build_oof_table(args.models)
    test = _build_test_table(args.models)
    print(f"  OOF rows: {len(oof)}  test rows: {len(test)}")

    # --- Per-leg iso-recal on the calibration set (OOF) -----------------
    iso_enabled = not args.no_iso_recal
    print(f"\n[2/5] Iso-recal on OOF (enabled={iso_enabled})...")
    oof, test = _per_leg_iso_recal(oof, test, args.models, enable=iso_enabled)

    # Sanity: report base nMAE (calibrated 50/50 blend per model) on OOF.
    print("\n[3/5] Per-model OOF nMAE (after iso-recal, 50/50 leg blend):")
    for tag in args.models:
        blend = 0.5 * oof[f"{tag}_cal_mw_mw"] + 0.5 * oof[f"{tag}_cal_cf_mw"]
        blend = np.clip(blend, 0.0, CAPACITY_MW)
        n = float(normalized_mae(oof["target_mw"].to_numpy(), blend))
        print(f"  {tag}: {n:.4f}%")

    # --- Ridge stack: leave-one-fold-out search over alpha --------------
    include_extras = not args.no_extras
    print(f"\n[4/5] Ridge LOO evaluation (extras={include_extras})...")
    best_alpha, best_nmae = _ridge_loo_eval(
        oof, args.models, alphas=args.alphas, include_extras=include_extras,
    )
    print(f"\n  Best alpha: {best_alpha}  LOO nMAE: {best_nmae:.4f}%")

    # --- Final fit on ALL OOF, apply to test ---------------------------
    print("\n[5/5] Final fit + test inference...")
    X_train = _build_stacker_X(oof, args.models, include_extras=include_extras)
    y_train = oof["target_mw"].to_numpy()
    scaler = StandardScaler().fit(X_train)
    ridge = Ridge(alpha=best_alpha).fit(scaler.transform(X_train), y_train)

    print(f"  Ridge coef (after StandardScaler):")
    feat_names = []
    for tag in args.models:
        feat_names.extend([f"{tag}_cal_cf", f"{tag}_cal_mw"])
    if include_extras:
        feat_names.extend(["ws_120", "ws_120_sq", "hour_sin", "hour_cos", "active_turbines"])
    for name, coef in zip(feat_names, ridge.coef_, strict=True):
        print(f"    {name:<22s}  {coef:+8.4f}")
    print(f"    intercept             {ridge.intercept_:+8.4f}")

    # Inference on test.
    X_test = _build_stacker_X(test, args.models, include_extras=include_extras)
    final_mw = np.clip(ridge.predict(scaler.transform(X_test)), 0.0, CAPACITY_MW)

    order = test["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = test[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    df_valid = load_valid_features(VALID_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)

    print(f"\n  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   "
          f"Std: {final_mw.std():.2f} MW   "
          f"Range: [{final_mw.min():.2f}, {final_mw.max():.2f}]")


if __name__ == "__main__":
    main()
