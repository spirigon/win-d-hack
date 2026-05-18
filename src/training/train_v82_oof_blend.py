"""V82: OOF stacking blend of V75 + V79 + V80.

Optimizes per-version CF blend weights on Fold-5 OOF (most similar to test:
Jan-Mar 2025 ≈ Jan-Mar 2026).  Applies to per-fold test predictions from each
version's saved test parquet.

Models:
  v75 — LGB K=80, baseline (LB 7.43%)
  v79 — LGB+XGB 50/50 (LB 7.54%) — structural diversity from XGBoost
  v80 — LGB K=120 (LB 7.505%) — slightly different feature set

Outputs:
    submissions/archive/v82.0_oof_blend.csv

Usage:
    python -m src.training.train_v82_oof_blend
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission
from src.training.train_v32_era5v2 import _from_cf

# Input parquets
OOF_PATHS = {
    "v75": _ROOT / "data" / "processed" / "v75_oof.parquet",
    "v79": _ROOT / "data" / "processed" / "v79_oof.parquet",
    "v80": _ROOT / "data" / "processed" / "v80_oof.parquet",
}
TEST_PATHS = {
    "v75": _ROOT / "data" / "processed" / "v75_test.parquet",
    "v79": _ROOT / "data" / "processed" / "v79_test.parquet",
    "v80": _ROOT / "data" / "processed" / "v80_test.parquet",
}
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v82.0_oof_blend.csv"
VALID_PATH  = _ROOT / "data" / "raw" / "valid_features.csv"


def _load_oof_cf(path: Path, key: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    col = "cf" if "cf" in df.columns and key == "v75" else "pred_cf"
    return pd.DataFrame({
        "fold": df["fold"].to_numpy(),
        "ts": df["ts"].to_numpy(),
        "target_mw": df["target_mw"].to_numpy(),
        "active_turbines": df["active_turbines"].to_numpy(),
        f"pred_cf_{key}": df[col].to_numpy(),
    })


def main() -> None:
    print("=" * 72)
    print("V82: OOF blend — V75 + V79 + V80  (optimize on Fold-5)")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # Load and merge OOF predictions
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading OOF predictions...")
    oofs = {k: _load_oof_cf(p, k) for k, p in OOF_PATHS.items()}

    # Merge on (fold, ts)
    merged = oofs["v75"]
    for k in ["v79", "v80"]:
        merged = merged.merge(
            oofs[k][["fold", "ts", f"pred_cf_{k}"]],
            on=["fold", "ts"], how="inner",
        )
    print(f"  Merged OOF rows: {len(merged)}  (expected 6176)")

    for k in ["v75", "v79", "v80"]:
        pred_mw = np.clip(
            _from_cf(merged[f"pred_cf_{k}"].to_numpy(), merged["active_turbines"].to_numpy()),
            0, CAPACITY_MW,
        )
        nmae = float(normalized_mae(merged["target_mw"].to_numpy(), pred_mw))
        f5 = merged[merged.fold == 5]
        pred_mw5 = np.clip(
            _from_cf(f5[f"pred_cf_{k}"].to_numpy(), f5["active_turbines"].to_numpy()),
            0, CAPACITY_MW,
        )
        nmae5 = float(normalized_mae(f5["target_mw"].to_numpy(), pred_mw5))
        print(f"  {k}: all-fold nMAE={nmae:.4f}%  fold5 nMAE={nmae5:.4f}%")

    # ------------------------------------------------------------------ #
    # Optimize blend weights on Fold-5
    # ------------------------------------------------------------------ #
    print("\n[2/4] Optimizing blend weights (Fold-5 nMAE)...")
    f5 = merged[merged.fold == 5].reset_index(drop=True)
    active5 = f5["active_turbines"].to_numpy()
    target5 = f5["target_mw"].to_numpy()

    keys = ["v75", "v79", "v80"]
    cf_stack = np.stack([f5[f"pred_cf_{k}"].to_numpy() for k in keys], axis=1)  # (N, 3)

    def objective(w):
        blend_cf = cf_stack @ w
        pred_mw = np.clip(_from_cf(blend_cf, active5), 0, CAPACITY_MW)
        return float(normalized_mae(target5, pred_mw))

    # Grid search to find good starting point
    best_val, best_w = 999.0, np.array([1.0, 0.0, 0.0])
    for a in np.arange(0.0, 1.05, 0.1):
        for b in np.arange(0.0, 1.05 - a, 0.1):
            c = 1.0 - a - b
            if c < -1e-9:
                continue
            w = np.array([a, b, c])
            v = objective(w)
            if v < best_val:
                best_val, best_w = v, w

    # Refine with scipy minimize (Dirichlet simplex constraint)
    def neg_obj(x):
        w = np.exp(x) / np.exp(x).sum()
        return objective(w)

    x0 = np.log(best_w + 1e-6)
    res = minimize(neg_obj, x0, method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-6})
    final_w = np.exp(res.x) / np.exp(res.x).sum()
    final_val = objective(final_w)

    print(f"  Best weights: " + "  ".join(f"{k}={final_w[i]:.3f}" for i, k in enumerate(keys)))
    print(f"  Blend Fold-5 nMAE: {final_val:.4f}%  "
          f"(vs v75 alone: {objective(np.array([1.,0.,0.])):.4f}%)")

    # Also report all-fold blend nMAE
    active_all = merged["active_turbines"].to_numpy()
    target_all  = merged["target_mw"].to_numpy()
    cf_all = np.stack([merged[f"pred_cf_{k}"].to_numpy() for k in keys], axis=1)
    blend_all = cf_all @ final_w
    pred_all = np.clip(_from_cf(blend_all, active_all), 0, CAPACITY_MW)
    print(f"  Blend all-fold nMAE: {float(normalized_mae(target_all, pred_all)):.4f}%")

    # ------------------------------------------------------------------ #
    # Build blended test submission
    # ------------------------------------------------------------------ #
    print("\n[3/4] Building test submission...")
    tests = {k: pd.read_parquet(p) for k, p in TEST_PATHS.items()}

    # Average fold-level CF predictions within each version
    def avg_test_cf(df: pd.DataFrame) -> np.ndarray:
        cf_cols = [c for c in df.columns if c.startswith("test_cf_fold")]
        return df[cf_cols].mean(axis=1).to_numpy()

    t75 = tests["v75"]
    test_cf_versions = np.stack(
        [avg_test_cf(tests[k]) for k in keys], axis=1
    )  # (2126, 3)
    blend_test_cf = test_cf_versions @ final_w
    active_test = t75["active_turbines"].to_numpy(dtype=np.float32)
    final_mw = np.clip(_from_cf(blend_test_cf, active_test), 0.0, CAPACITY_MW).astype(np.float64)

    order = t75["_submission_row"].to_numpy().astype(int)
    po = np.empty_like(final_mw)
    po[order] = final_mw
    ts_arr = t75[TIMESTAMP_COL].to_numpy()
    ts_po = np.empty_like(ts_arr)
    ts_po[order] = ts_arr

    # ------------------------------------------------------------------ #
    # Write submission
    # ------------------------------------------------------------------ #
    print("[4/4] Writing submission...")
    n_valid = len(pd.read_csv(VALID_PATH))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(po, OUTPUT_PATH, expected_rows=n_valid, timestamps=ts_po)
    print(f"  Submission saved: {OUTPUT_PATH}")
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW")
    print(f"  Blend weights: " + "  ".join(f"{k}={final_w[i]:.3f}" for i, k in enumerate(keys)))


if __name__ == "__main__":
    main()
