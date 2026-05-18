"""V104: Ridge-stacking meta-model over v99/v100/v101/v102/v103 OOF predictions.

Each base model contributes an OOF prediction column.  A ridge regression
meta-model is fit on the stacked OOF features → target_mw, then applied to
the corresponding test submission CSVs (which cover the same timestamps).

Design:
  - Meta-features: pred_blend_mw from each available base model (v99-v103)
  - Meta-target: target_mw (actual MW)
  - Meta-model: Ridge regression (low variance, extracts optimal linear blend)
  - No temporal leakage: each base-model OOF row was held-out during training
  - Test predictions: apply ridge coefficients to base model test CSV predictions

Outputs:
    submissions/archive/v104.0_stack.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

PROCESSED_DIR  = _ROOT / "data" / "processed"
SUBMISSIONS_DIR = _ROOT / "submissions" / "archive"
OUTPUT_PATH    = SUBMISSIONS_DIR / "v104.0_stack.csv"

# Base models to stack — will use whichever OOF parquets exist
BASE_VERSIONS = ["v99", "v100", "v101", "v102", "v103"]
# Map version → submission CSV filename
SUBMISSION_NAMES = {
    "v99":  "v99.0_corr_dedup.csv",
    "v100": "v100.0_dir_disagree.csv",
    "v101": "v101.0_catboost.csv",   # catboost+lgbm blend
    "v102": "v102.0_multifold_probe.csv",
    "v103": "v103.0_mae_obj.csv",
}

RIDGE_ALPHA = 1.0  # ridge regularization


def _load_oof(version: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{version}_oof.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df.rename(columns={"pred_blend_mw": f"{version}_pred"})
    return df[["fold", "ts", "target_mw", f"{version}_pred"]]


def _load_test_preds(version: str) -> np.ndarray | None:
    path = SUBMISSIONS_DIR / SUBMISSION_NAMES[version]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Submission format: 2 columns — timestamp and predicted_power
    pred_col = [c for c in df.columns if c != df.columns[0]][0]
    return df[pred_col].to_numpy(dtype=np.float64)


def main() -> None:
    print("=" * 72)
    print("V104: Ridge stacking meta-model")
    print("=" * 72)

    # Load all available OOF predictions
    oof_list = []
    available = []
    for ver in BASE_VERSIONS:
        df = _load_oof(ver)
        if df is not None:
            oof_list.append(df)
            available.append(ver)
            print(f"  Loaded OOF: {ver} ({len(df)} rows)")
        else:
            print(f"  Missing OOF: {ver} — skipped")

    if len(available) < 2:
        print("ERROR: need at least 2 base models with OOF predictions")
        return

    # Merge all OOF on shared columns
    base = oof_list[0][["fold", "ts", "target_mw"]].copy()
    for df in oof_list:
        ver_col = [c for c in df.columns if c.endswith("_pred")][0]
        base = base.merge(df[["fold", "ts", ver_col]], on=["fold", "ts"], how="inner")

    pred_cols = [f"{v}_pred" for v in available]
    X = base[pred_cols].to_numpy(dtype=np.float64)
    y = base["target_mw"].to_numpy(dtype=np.float64)

    print(f"\n  Stacking dataset: {len(base)} rows, {len(pred_cols)} base models")

    # Fit ridge on full OOF (no leakage since all are held-out predictions)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(X_scaled, y)

    # OOF meta-prediction for diagnostics
    oof_pred = np.clip(ridge.predict(X_scaled), 0, None)
    oof_nmae = float(normalized_mae(y, oof_pred))
    print(f"\n  Meta-model OOF nMAE (train): {oof_nmae:.4f}%  (optimistic — same data used to fit)")

    # Per-fold breakdown
    for fid in sorted(base["fold"].unique()):
        mask = base["fold"] == fid
        sub_pred = np.clip(ridge.predict(X_scaled[mask]), 0, None)
        sub_nmae = float(normalized_mae(y[mask], sub_pred))
        print(f"    Fold {fid}: {sub_nmae:.4f}%")

    # Print coefficients for interpretability
    print("\n  Ridge coefficients (before scaling):")
    for ver, coef in zip(available, ridge.coef_):
        print(f"    {ver}: {coef:.4f}")
    print(f"    intercept: {ridge.intercept_:.4f}")

    # Load test predictions and apply ridge
    print("\n  Loading test predictions...")
    test_preds = {}
    for ver in available:
        preds = _load_test_preds(ver)
        if preds is not None:
            test_preds[ver] = preds
            print(f"    {ver}: {len(preds)} rows, mean={preds.mean():.2f} MW")
        else:
            print(f"    {ver}: submission CSV not found — using equal fallback")

    if len(test_preds) < len(available):
        print("WARNING: some test CSVs missing; filling with available mean")

    # Build test feature matrix
    n_test = max(len(v) for v in test_preds.values())
    X_test = np.zeros((n_test, len(available)), dtype=np.float64)
    for j, ver in enumerate(available):
        if ver in test_preds:
            X_test[:, j] = test_preds[ver]
        else:
            # Fill with column mean from OOF (neutral imputation)
            X_test[:, j] = X[:, j].mean()

    X_test_scaled = scaler.transform(X_test)
    final_mw = np.clip(ridge.predict(X_test_scaled), 0.0, None)

    print(f"\n  Final submission: mean={final_mw.mean():.2f} MW  std={final_mw.std():.2f} MW")

    # Load one submission CSV to get timestamps/row order
    ref_ver = next(v for v in available if v in test_preds)
    ref_path = SUBMISSIONS_DIR / SUBMISSION_NAMES[ref_ver]
    ref_df = pd.read_csv(ref_path)
    ts_col = ref_df.columns[0]
    timestamps = pd.to_datetime(ref_df[ts_col]).to_numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(final_mw, OUTPUT_PATH, expected_rows=n_test, timestamps=timestamps)
    print(f"  Submission saved: {OUTPUT_PATH}")

    # Also report pairwise OOF correlation between base models
    print("\n  Base model OOF pairwise correlation:")
    for i, v1 in enumerate(available):
        for j, v2 in enumerate(available):
            if j <= i:
                continue
            r = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
            print(f"    {v1} vs {v2}: r={r:.4f}")


if __name__ == "__main__":
    main()
