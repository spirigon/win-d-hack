"""V37: Simple non-negative blend of v32 / v34 / v35 50/50 leg outputs.

The v36 ridge stack overfit Fold-3 (had +12 / −10 offsetting coefficients
on highly-correlated leg pairs). A safer alternative: take each model's
50/50 mw+cf blend (the standalone submission output), then blend those
THREE summary predictions with non-negative weights that sum to 1.

Search:
    grid over (w_v32, w_v34, w_v35) on the simplex.
    For each grid point, compute mean OOF nMAE across folds.

Output:
    submissions/archive/v37.0_simple_blend.csv

Usage:

    python -m src.training.train_v37_simple_blend
    python -m src.training.train_v37_simple_blend --models v32 v34
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OOF_BASE = _ROOT / "data" / "processed"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v37.0_simple_blend.csv"

FOLD_IDS = [3, 4, 5]
TURBINE_RATED_MW = 3.465


def _model_blend_oof(tag: str) -> pd.DataFrame:
    """Load OOF and produce per-row 50/50 mw+cf blend in MW units."""
    oof = pd.read_parquet(OOF_BASE / f"{tag}_oof.parquet")
    blend = 0.5 * oof["pred_mw_mw"].to_numpy() + 0.5 * oof["pred_cf_mw"].to_numpy()
    blend = np.clip(blend, 0.0, CAPACITY_MW)
    return pd.DataFrame({
        "fold": oof["fold"].astype(int),
        "ts": oof["ts"],
        "target_mw": oof["target_mw"].to_numpy(),
        f"{tag}_blend_mw": blend,
    })


def _model_blend_test(tag: str) -> pd.DataFrame:
    """Load test parquet and produce per-row 50/50 mw+cf blend in MW units.

    Per-fold CFs (range 0-1) are first converted to MW using
    active_turbines × 3.465, then averaged across the CV-bag folds.
    """
    df = pd.read_parquet(OOF_BASE / f"{tag}_test.parquet")
    active = df["active_turbines"].to_numpy(dtype=np.float32)

    cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS if f"test_cf_fold{i}" in df.columns]
    mw_cols = [f"test_mw_fold{i}" for i in FOLD_IDS if f"test_mw_fold{i}" in df.columns]

    cf_mw_per_fold = np.column_stack([
        np.clip(np.clip(df[c].to_numpy(), 0.0, 1.0) * active * TURBINE_RATED_MW, 0.0, CAPACITY_MW)
        for c in cf_cols
    ])
    avg_cf_mw = cf_mw_per_fold.mean(axis=1)
    avg_mw_mw = np.clip(df[mw_cols].mean(axis=1).to_numpy(), 0.0, CAPACITY_MW)
    blend = np.clip(0.5 * avg_mw_mw + 0.5 * avg_cf_mw, 0.0, CAPACITY_MW)

    return pd.DataFrame({
        TIMESTAMP_COL: df[TIMESTAMP_COL],
        "_submission_row": df["_submission_row"].astype(int),
        f"{tag}_blend_mw": blend,
    })


def _grid_simplex(n_models: int, step: float = 0.1):
    """Yield non-negative weight vectors that sum to 1, with grid step."""
    if n_models == 1:
        yield (1.0,)
        return
    n_steps = int(round(1.0 / step))
    for combo in product(range(n_steps + 1), repeat=n_models - 1):
        if sum(combo) > n_steps:
            continue
        ws = [c * step for c in combo]
        ws.append(1.0 - sum(ws))
        if ws[-1] < -1e-9:
            continue
        ws[-1] = max(0.0, ws[-1])
        yield tuple(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v32", "v34", "v35"])
    ap.add_argument("--step", type=float, default=0.05,
                    help="grid step for the simplex search (default: 0.05)")
    args = ap.parse_args()

    print("=" * 72)
    print(f"V37: Simple non-negative blend over {args.models}")
    print("=" * 72)

    # --- Load and join OOF blends ----------------------------------------
    oof = None
    for tag in args.models:
        df = _model_blend_oof(tag)
        oof = df if oof is None else oof.merge(
            df.drop(columns=["target_mw"]), on=["fold", "ts"], how="inner",
        )
    print(f"\nOOF rows: {len(oof)}")
    for tag in args.models:
        n = float(normalized_mae(oof["target_mw"].to_numpy(), oof[f"{tag}_blend_mw"].to_numpy()))
        print(f"  {tag} 50/50-blend OOF nMAE: {n:.4f}%")

    # --- Grid search on simplex -----------------------------------------
    blend_arr = np.column_stack([oof[f"{tag}_blend_mw"].to_numpy() for tag in args.models])
    folds = oof["fold"].to_numpy()
    target = oof["target_mw"].to_numpy()
    fold_ids = sorted(np.unique(folds))

    print(f"\nGrid search (step={args.step})...")
    best_w = None
    best_nmae_mean = np.inf
    best_per_fold = None

    for ws in _grid_simplex(len(args.models), step=args.step):
        ws_arr = np.array(ws, dtype=np.float64)
        preds = np.clip(blend_arr @ ws_arr, 0.0, CAPACITY_MW)
        per_fold = []
        for fid in fold_ids:
            mask = folds == fid
            per_fold.append(float(normalized_mae(target[mask], preds[mask])))
        mean_nmae = float(np.mean(per_fold))
        if mean_nmae < best_nmae_mean:
            best_nmae_mean = mean_nmae
            best_w = ws
            best_per_fold = per_fold

    print(f"\nBest weights: {dict(zip(args.models, [round(w, 3) for w in best_w], strict=True))}")
    print(f"  Per-fold nMAE: {[f'{n:.4f}' for n in best_per_fold]}")
    print(f"  Mean nMAE: {best_nmae_mean:.4f}%")

    # Also report Fold-5 only (the most relevant for Q1 2026).
    f5_idx = fold_ids.index(5) if 5 in fold_ids else -1
    if f5_idx >= 0:
        print(f"  Fold-5 nMAE: {best_per_fold[f5_idx]:.4f}%")

    # --- Apply best weights to test --------------------------------------
    test = None
    for tag in args.models:
        df = _model_blend_test(tag)
        test = df if test is None else test.merge(
            df.drop(columns=["_submission_row"]), on=TIMESTAMP_COL, how="inner",
        )
    test_blend = np.column_stack([test[f"{tag}_blend_mw"].to_numpy() for tag in args.models])
    final_mw = np.clip(test_blend @ np.array(best_w, dtype=np.float64), 0.0, CAPACITY_MW)

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
          f"Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
