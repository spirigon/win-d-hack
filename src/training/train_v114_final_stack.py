"""V114: Final best-of-all Ridge stacking — includes v113 (K=120 byte-dedup).

Runs after v113 completes. Re-stacks all available OOF predictions
including the new v113 model. Identical logic to v112 but with v113 added
to the candidate pool.

Outputs:
    data/processed/v114_oof.parquet
    submissions/archive/v114.0_final_stack.csv
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

PROCESSED_DIR   = _ROOT / "data" / "processed"
SUBMISSIONS_DIR = _ROOT / "submissions" / "archive"
OUTPUT_PATH     = SUBMISSIONS_DIR / "v114.0_final_stack.csv"
OOF_PATH        = PROCESSED_DIR / "v114_oof.parquet"

CANDIDATE_VERSIONS = [
    "v97", "v97b", "v99", "v100", "v102", "v103",
    "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v113",
]

SUBMISSION_MAP: dict[str, str] = {
    "v97":  "v97.0_dedup_gem.csv",
    "v97b": "v97b.0_cfonly.csv",
    "v99":  "v99.0_corr_dedup.csv",
    "v100": "v100.0_dir_disagree.csv",
    "v101": "v101.0_catboost.csv",
    "v102": "v102.0_multifold_probe.csv",
    "v103": "v103.0_mae_obj.csv",
    "v104": "v104.0_stack.csv",
    "v105": "v105.0_k120.csv",
    "v106": "v106.0_pcslope.csv",
    "v107": "v107.0_time_weight.csv",
    "v108": "v108.0_mlp.csv",
    "v109": "v109.0_turbulence.csv",
    "v110": "v110.0_xgboost.csv",
    "v111": "v111.0_optuna_hpo.csv",
    "v112": "v112.0_best_stack.csv",
    "v113": "v113.0_k120_bytededup.csv",
}

QUALITY_THRESHOLD = 7.60  # F5 nMAE — excludes poor models
RIDGE_ALPHA = 1.0


def _load_oof(version: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{version}_oof.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "pred_blend_mw" not in df.columns:
        if "pred_mlp_mw" in df.columns:
            df = df.rename(columns={"pred_mlp_mw": "pred_blend_mw"})
        else:
            return None
    return df[["fold", "ts", "target_mw", "pred_blend_mw"]].rename(
        columns={"pred_blend_mw": f"{version}_pred"}
    )


def _f5_nmae(df_oof: pd.DataFrame, version: str) -> float:
    sub = df_oof[df_oof["fold"] == 5]
    if len(sub) == 0:
        return 999.0
    return float(normalized_mae(sub["target_mw"].to_numpy(), sub[f"{version}_pred"].to_numpy()))


def _load_test_preds(version: str) -> np.ndarray | None:
    fname = SUBMISSION_MAP.get(version)
    if fname is None:
        return None
    path = SUBMISSIONS_DIR / fname
    if not path.exists():
        return None
    df = pd.read_csv(path)
    pred_col = [c for c in df.columns if c != df.columns[0]][0]
    return df[pred_col].to_numpy(dtype=np.float64)


def main() -> None:
    print("=" * 72)
    print("V114: Final Ridge stacking (all experiments incl. v113)")
    print("=" * 72)

    oof_list, available = [], []
    quality: dict[str, float] = {}

    print("\n  Scanning candidate versions...")
    for ver in CANDIDATE_VERSIONS:
        df = _load_oof(ver)
        if df is None:
            print(f"    {ver}: OOF not found — skipped")
            continue
        f5 = _f5_nmae(df, ver)
        if f5 >= QUALITY_THRESHOLD:
            print(f"    {ver}: F5={f5:.4f}% — EXCLUDED (above threshold {QUALITY_THRESHOLD}%)")
            continue
        oof_list.append(df)
        available.append(ver)
        quality[ver] = f5
        print(f"    {ver}: F5={f5:.4f}% — included")

    if len(available) < 2:
        print(f"ERROR: need at least 2 models below F5 threshold {QUALITY_THRESHOLD}%")
        return

    print(f"\n  Using {len(available)} base models: {available}")

    base = oof_list[0][["fold", "ts", "target_mw"]].copy()
    for df in oof_list:
        ver_col = [c for c in df.columns if c.endswith("_pred")][0]
        base = base.merge(df[["fold", "ts", ver_col]], on=["fold", "ts"], how="inner")

    pred_cols = [f"{v}_pred" for v in available]
    X = base[pred_cols].to_numpy(dtype=np.float64)
    y = base["target_mw"].to_numpy(dtype=np.float64)

    print(f"\n  Pairwise F5 Pearson correlations:")
    sub5 = base[base["fold"] == 5]
    X5 = sub5[pred_cols].to_numpy(dtype=np.float64)
    if len(X5) > 10:
        corr = np.corrcoef(X5.T)
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                print(f"    {available[i]} vs {available[j]}: r={corr[i, j]:.4f}")

    folds = sorted(base["fold"].unique())
    meta_preds = np.zeros(len(base), dtype=np.float64)
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    ridge_full = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge_full.fit(X_scaled_full, y)

    print(f"\n  Leave-fold-out meta-model OOF:")
    for target_fold in folds:
        val_mask = base["fold"] == target_fold
        cal_mask = base["fold"] != target_fold
        X_cal, y_cal = X[cal_mask], y[cal_mask]
        X_val = X[val_mask]
        sc = StandardScaler()
        X_cal_s = sc.fit_transform(X_cal)
        X_val_s = sc.transform(X_val)
        ridge_fold = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        ridge_fold.fit(X_cal_s, y_cal)
        meta_preds[val_mask] = np.clip(ridge_fold.predict(X_val_s), 0, None)

    oof_nmae = float(normalized_mae(y, meta_preds))
    print(f"  Overall leave-fold-out OOF nMAE: {oof_nmae:.4f}%")
    for fid in folds:
        mask = base["fold"] == fid
        fid_nmae = float(normalized_mae(y[mask], meta_preds[mask]))
        print(f"    Fold {fid}: {fid_nmae:.4f}%")

    print(f"\n  Ridge coefficients (full model, alpha={RIDGE_ALPHA}):")
    for ver, coef in zip(available, ridge_full.coef_):
        print(f"    {ver}: {coef:.4f}  (F5={quality[ver]:.4f}%)")
    print(f"    intercept: {ridge_full.intercept_:.4f}")

    oof_df = pd.DataFrame({
        "fold":          base["fold"].to_numpy(),
        "ts":            base["ts"].to_numpy(),
        "target_mw":     y,
        "pred_blend_mw": meta_preds,
    })
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")

    print("\n  Loading test predictions...")
    test_preds: dict[str, np.ndarray] = {}
    n_test = 2126
    for ver in available:
        preds = _load_test_preds(ver)
        if preds is not None:
            test_preds[ver] = preds
            n_test = len(preds)
            print(f"    {ver}: {len(preds)} rows, mean={preds.mean():.2f} MW")
        else:
            print(f"    {ver}: submission CSV not found — filling with column mean")

    X_test = np.zeros((n_test, len(available)), dtype=np.float64)
    for j, ver in enumerate(available):
        if ver in test_preds:
            X_test[:, j] = test_preds[ver]
        else:
            X_test[:, j] = X[:, j].mean()

    X_test_scaled = scaler_full.transform(X_test)
    final_mw = np.clip(ridge_full.predict(X_test_scaled), 0.0, None)
    print(f"\n  Final: mean={final_mw.mean():.2f} MW  std={final_mw.std():.2f} MW")

    ref_ver = next(v for v in available if v in test_preds)
    ref_sub = pd.read_csv(SUBMISSIONS_DIR / SUBMISSION_MAP[ref_ver])
    timestamps = ref_sub.iloc[:, 0].to_numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(final_mw, OUTPUT_PATH, expected_rows=n_test, timestamps=timestamps)
    print(f"  Submission saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
