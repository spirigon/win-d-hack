"""V41: Fold-5-tuned blend of v32 / v34 / v40 CF-only OOF.

The structural features in v40 helped Fold-3/4 but didn't move Fold-5
(7.574 vs v32's 7.537). However the per-fold predictions might still
combine well — v40 makes different errors on Fold-5 from v32 even when
the magnitudes are similar.

This script:
    1. Reads OOF + test predictions for v32, v34, v40 (CF leg only).
    2. Grid-searches non-negative weights summing to 1 on Fold-5 OOF.
    3. Applies the best weights to the averaged test predictions.

Output:
    submissions/archive/v41.0_blend_v32_v34_v40.csv
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission

VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OOF_BASE = _ROOT / "data" / "processed"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v41.0_blend_v32_v34_v40.csv"

MODELS = ["v32", "v34", "v40"]
TURBINE_RATED_MW = 3.465
FOLD_IDS = [3, 4, 5]


def _model_oof_cf_mw(tag: str) -> pd.DataFrame:
    oof = pd.read_parquet(OOF_BASE / f"{tag}_oof.parquet")
    return pd.DataFrame({
        "fold": oof["fold"].astype(int),
        "ts": oof["ts"],
        "target_mw": oof["target_mw"].to_numpy(),
        f"{tag}_cf_mw": oof["pred_cf_mw"].to_numpy(),
    })


def _model_test_cf_mw(tag: str) -> pd.DataFrame:
    df = pd.read_parquet(OOF_BASE / f"{tag}_test.parquet")
    active = df["active_turbines"].to_numpy(dtype=np.float32)
    cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS if f"test_cf_fold{i}" in df.columns]
    cf_mw_per_fold = np.column_stack([
        np.clip(np.clip(df[c].to_numpy(), 0.0, 1.0) * active * TURBINE_RATED_MW,
                0.0, CAPACITY_MW)
        for c in cf_cols
    ])
    return pd.DataFrame({
        TIMESTAMP_COL: df[TIMESTAMP_COL],
        "_submission_row": df["_submission_row"].astype(int),
        f"{tag}_cf_mw": cf_mw_per_fold.mean(axis=1).astype(np.float64),
    })


def _simplex(n: int, step: float = 0.05):
    n_steps = int(round(1.0 / step))
    for combo in product(range(n_steps + 1), repeat=n - 1):
        if sum(combo) > n_steps:
            continue
        ws = [c * step for c in combo]
        ws.append(1.0 - sum(ws))
        if ws[-1] < -1e-9:
            continue
        ws[-1] = max(0.0, ws[-1])
        yield tuple(ws)


def main():
    # --- Load OOF ---
    oof = None
    for tag in MODELS:
        df = _model_oof_cf_mw(tag)
        oof = df if oof is None else oof.merge(
            df.drop(columns=["target_mw"]), on=["fold", "ts"], how="inner",
        )
    print(f"OOF rows: {len(oof)}")
    print("\nPer-model CF-only OOF nMAE:")
    target = oof["target_mw"].to_numpy()
    folds = oof["fold"].to_numpy()
    for tag in MODELS:
        n_all = float(normalized_mae(target, oof[f"{tag}_cf_mw"].to_numpy()))
        n_f5 = float(normalized_mae(target[folds == 5], oof.loc[folds == 5, f"{tag}_cf_mw"].to_numpy()))
        print(f"  {tag}: all_folds={n_all:.4f}%   Fold-5={n_f5:.4f}%")

    # --- Grid search on Fold-5 only ---
    pred_mat = np.column_stack([oof[f"{tag}_cf_mw"].to_numpy() for tag in MODELS])
    f5_mask = folds == 5

    best_f5 = np.inf
    best_w = None
    best_per_fold = None
    for ws in _simplex(len(MODELS), step=0.05):
        ws_arr = np.array(ws, dtype=np.float64)
        preds = np.clip(pred_mat @ ws_arr, 0.0, CAPACITY_MW)
        f5 = float(normalized_mae(target[f5_mask], preds[f5_mask]))
        if f5 < best_f5:
            best_f5 = f5
            best_w = ws
            best_per_fold = [
                float(normalized_mae(target[folds == fid], preds[folds == fid]))
                for fid in FOLD_IDS
            ]
    print(f"\nBest blend by Fold-5 nMAE:")
    print(f"  Weights: {dict(zip(MODELS, [round(w, 3) for w in best_w]))}")
    print(f"  Per-fold: {[f'{n:.4f}' for n in best_per_fold]}")
    print(f"  Mean of folds: {np.mean(best_per_fold):.4f}%")
    print(f"  Fold-5: {best_f5:.4f}%")

    # --- Apply to test ---
    test = None
    for tag in MODELS:
        df = _model_test_cf_mw(tag)
        test = df if test is None else test.merge(
            df.drop(columns=["_submission_row"]), on=TIMESTAMP_COL, how="inner",
        )
    test_mat = np.column_stack([test[f"{tag}_cf_mw"].to_numpy() for tag in MODELS])
    final_mw = np.clip(test_mat @ np.array(best_w, dtype=np.float64), 0.0, CAPACITY_MW)

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
    print(f"  Mean: {final_mw.mean():.2f} MW   Std: {final_mw.std():.2f} MW")


if __name__ == "__main__":
    main()
