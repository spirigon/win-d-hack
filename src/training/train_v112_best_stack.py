"""V112: Best-of-all Ridge stacking — runs after all individual experiments complete.

Dynamically loads all available OOF parquets, filters to models with
Fold-5 CF nMAE < QUALITY_THRESHOLD, then fits leave-fold-out Ridge stacking
on the best performers. Better than V104 because:
  - Uses all experiments (v97b through v111, not just v99-v103)
  - Includes v97b (byte-dedup, best single model) in the pool
  - Filters out weak models (CatBoost-heavy: v101 blend)
  - Leave-fold-out meta-model avoids over-fitting

Outputs:
    data/processed/v112_oof.parquet
    submissions/archive/v112.0_best_stack.csv
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
OUTPUT_PATH       = SUBMISSIONS_DIR / "v112.0_best_stack.csv"
OUTPUT_BLEND_PATH = SUBMISSIONS_DIR / "v112.1_stack_blend50.csv"
OOF_PATH          = PROCESSED_DIR / "v112_oof.parquet"

# All candidate versions — loaded dynamically
CANDIDATE_VERSIONS = [
    "v97", "v97b", "v99", "v100", "v102", "v103",
    "v105", "v106", "v107", "v108", "v109", "v110", "v111",
]

# CF-only submissions (used for OOF-consistent stacking)
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
}

# Blend-50/50 submissions — use where available; fall back to CF-only
# LB shows blend50 beats CF-only by ~0.04pp (v97b.0 CF=7.45% vs v97b.1 blend=7.415%)
BLEND50_MAP: dict[str, str] = {
    "v97":  "v97.0_dedup_gem.csv",       # v97 already used 50/50 blend
    "v97b": "v97b.1_blend50.csv",
    "v109": "v109.1_blend50.csv",        # written if v109 re-run; else falls back to CF-only
    "v111": "v111.1_blend50.csv",
}

# Only include models with F5 blend nMAE below this threshold
QUALITY_THRESHOLD = 7.60  # pp — excludes v101 (7.64%) and poor models

RIDGE_ALPHA = 1.0


def _load_oof(version: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{version}_oof.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Standardize: use pred_blend_mw column (all versions have this)
    if "pred_blend_mw" not in df.columns:
        if "pred_mlp_mw" in df.columns:  # v108 format
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
    print("V112: Best-of-all Ridge stacking")
    print("=" * 72)

    # Load all available OOF predictions
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

    # Merge OOF on (fold, ts)
    base = oof_list[0][["fold", "ts", "target_mw"]].copy()
    for df in oof_list:
        ver_col = [c for c in df.columns if c.endswith("_pred")][0]
        base = base.merge(df[["fold", "ts", ver_col]], on=["fold", "ts"], how="inner")

    pred_cols = [f"{v}_pred" for v in available]
    X = base[pred_cols].to_numpy(dtype=np.float64)
    y = base["target_mw"].to_numpy(dtype=np.float64)

    # Per-fold correlation matrix
    print(f"\n  Pairwise F5 Pearson correlations:")
    sub5 = base[base["fold"] == 5]
    X5 = sub5[pred_cols].to_numpy(dtype=np.float64)
    if len(X5) > 10:
        corr = np.corrcoef(X5.T)
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                print(f"    {available[i]} vs {available[j]}: r={corr[i, j]:.4f}")

    # Leave-fold-out Ridge stacking (proper OOF meta-predictions)
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

    # Print coefficients (full model trained on all data)
    print(f"\n  Ridge coefficients (full model, alpha={RIDGE_ALPHA}):")
    for ver, coef in zip(available, ridge_full.coef_):
        print(f"    {ver}: {coef:.4f}  (F5={quality[ver]:.4f}%)")
    print(f"    intercept: {ridge_full.intercept_:.4f}")

    # Save OOF
    oof_df = pd.DataFrame({
        "fold":        base["fold"].to_numpy(),
        "ts":          base["ts"].to_numpy(),
        "target_mw":   y,
        "pred_blend_mw": meta_preds,
    })
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")

    # Load test predictions and apply full Ridge
    print("\n  Loading test predictions...")
    test_preds: dict[str, np.ndarray] = {}
    n_test = 2126  # expected rows
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

    # Use first available test CSV for timestamps/row order
    ref_ver = next(v for v in available if v in test_preds)
    ref_sub = pd.read_csv(SUBMISSIONS_DIR / SUBMISSION_MAP[ref_ver])
    timestamps = ref_sub.iloc[:, 0].to_numpy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(final_mw, OUTPUT_PATH, expected_rows=n_test, timestamps=timestamps)
    print(f"  Submission saved: {OUTPUT_PATH}")

    # Blend50 stack: same Ridge weights, substitute blend50 CSVs where available
    # OOF is CF-only so Ridge weights are consistent; blend50 test preds shift LB ~-0.04pp
    print("\n  Building blend-50/50 stack submission...")
    test_preds_blend: dict[str, np.ndarray] = {}
    for ver in available:
        # Prefer blend50 CSV, fall back to CF-only
        blend_fname = BLEND50_MAP.get(ver, SUBMISSION_MAP.get(ver, ""))
        blend_path  = SUBMISSIONS_DIR / blend_fname if blend_fname else None
        if blend_path and blend_path.exists():
            df_b = pd.read_csv(blend_path)
            pred_col = [c for c in df_b.columns if c != df_b.columns[0]][0]
            test_preds_blend[ver] = df_b[pred_col].to_numpy(dtype=np.float64)
        elif ver in test_preds:
            test_preds_blend[ver] = test_preds[ver]  # fallback to CF-only

    X_test_b = np.zeros((n_test, len(available)), dtype=np.float64)
    for j, ver in enumerate(available):
        if ver in test_preds_blend:
            X_test_b[:, j] = test_preds_blend[ver]
        else:
            X_test_b[:, j] = X[:, j].mean()

    X_test_b_scaled = scaler_full.transform(X_test_b)
    final_mw_blend = np.clip(ridge_full.predict(X_test_b_scaled), 0.0, None)
    print(f"  Blend stack: mean={final_mw_blend.mean():.2f} MW  std={final_mw_blend.std():.2f} MW")
    OUTPUT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(final_mw_blend, OUTPUT_BLEND_PATH, expected_rows=n_test, timestamps=timestamps)
    print(f"  Submission saved: {OUTPUT_BLEND_PATH}")


if __name__ == "__main__":
    main()
