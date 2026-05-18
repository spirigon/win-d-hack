"""Find optimal v32/v34/v35 blend weights using ONLY Fold-5 OOF nMAE.

Rationale: Fold-3 is consistently a hard fold (>11% nMAE for all models)
and dominates the mean across folds. Q1 2026 is the same calendar quarter
as Fold-5, so Fold-5 OOF performance is a more honest target.

This script also tries the asymmetric blend (different weights for the
CF leg vs the MW leg) to see if there's free signal there.

No retrain — just reads existing OOF parquets.

Output: prints the best blend, and writes ``submissions/archive/v38.0_fold5_blend.csv``.
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
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v38.0_fold5_blend.csv"

FOLD_IDS = [3, 4, 5]
TURBINE_RATED_MW = 3.465
MODELS = ["v32", "v34", "v35"]


def _model_blend_oof(tag: str) -> pd.DataFrame:
    oof = pd.read_parquet(OOF_BASE / f"{tag}_oof.parquet")
    return pd.DataFrame({
        "fold": oof["fold"].astype(int),
        "ts": oof["ts"],
        "target_mw": oof["target_mw"].to_numpy(),
        f"{tag}_pred_cf_mw": oof["pred_cf_mw"].to_numpy(),
        f"{tag}_pred_mw_mw": oof["pred_mw_mw"].to_numpy(),
    })


def _model_test(tag: str) -> pd.DataFrame:
    df = pd.read_parquet(OOF_BASE / f"{tag}_test.parquet")
    active = df["active_turbines"].to_numpy(dtype=np.float32)
    cf_cols = [f"test_cf_fold{i}" for i in FOLD_IDS]
    mw_cols = [f"test_mw_fold{i}" for i in FOLD_IDS]
    cf_mw_per_fold = np.column_stack([
        np.clip(np.clip(df[c].to_numpy(), 0.0, 1.0) * active * TURBINE_RATED_MW, 0.0, CAPACITY_MW)
        for c in cf_cols
    ])
    avg_cf_mw = cf_mw_per_fold.mean(axis=1)
    avg_mw_mw = np.clip(df[mw_cols].mean(axis=1).to_numpy(), 0.0, CAPACITY_MW)
    return pd.DataFrame({
        TIMESTAMP_COL: df[TIMESTAMP_COL],
        "_submission_row": df["_submission_row"].astype(int),
        f"{tag}_pred_cf_mw": avg_cf_mw,
        f"{tag}_pred_mw_mw": avg_mw_mw,
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
    # --- Load OOF -------------------------------------------------------
    oof = None
    for tag in MODELS:
        df = _model_blend_oof(tag)
        oof = df if oof is None else oof.merge(
            df.drop(columns=["target_mw"]), on=["fold", "ts"], how="inner",
        )
    print(f"OOF rows: {len(oof)}")

    # Pre-compute per-row leg arrays.
    cf_mat = np.column_stack([oof[f"{tag}_pred_cf_mw"].to_numpy() for tag in MODELS])
    mw_mat = np.column_stack([oof[f"{tag}_pred_mw_mw"].to_numpy() for tag in MODELS])
    target = oof["target_mw"].to_numpy()
    folds = oof["fold"].to_numpy()
    f5_mask = folds == 5

    # --- Search 1: per-leg model weights × CF/MW mix --------------------
    # Joint over (w_cf simplex × w_mw simplex × leg_blend ∈ {0..1})
    print("\nSearching per-leg model weights × CF/MW mix...")
    best = None  # (nmae_f5, nmae_mean, w_cf, w_mw, alpha_mw)
    n_combos = 0
    for w_cf in _simplex(len(MODELS), step=0.1):
        for w_mw in _simplex(len(MODELS), step=0.1):
            cf_pred = cf_mat @ np.array(w_cf)
            mw_pred = mw_mat @ np.array(w_mw)
            for alpha in np.arange(0.0, 1.01, 0.1):
                blend = np.clip(alpha * mw_pred + (1 - alpha) * cf_pred, 0.0, CAPACITY_MW)
                f5 = float(normalized_mae(target[f5_mask], blend[f5_mask]))
                m_per_fold = [float(normalized_mae(target[folds == fid], blend[folds == fid]))
                              for fid in [3, 4, 5]]
                m_mean = float(np.mean(m_per_fold))
                if best is None or f5 < best[0]:
                    best = (f5, m_mean, w_cf, w_mw, alpha, m_per_fold)
                n_combos += 1
    print(f"  Searched {n_combos} combinations")
    f5, m_mean, w_cf, w_mw, alpha, per_fold = best
    print(f"\nBest by Fold-5 nMAE:")
    print(f"  CF leg weights: {dict(zip(MODELS, [round(w, 2) for w in w_cf]))}")
    print(f"  MW leg weights: {dict(zip(MODELS, [round(w, 2) for w in w_mw]))}")
    print(f"  MW share α    : {alpha:.2f}")
    print(f"  Per-fold nMAE: {[f'{n:.4f}' for n in per_fold]}")
    print(f"  Fold-5 nMAE  : {f5:.4f}%")
    print(f"  Mean nMAE    : {m_mean:.4f}%")

    # --- Apply on test ---------------------------------------------------
    test = None
    for tag in MODELS:
        df = _model_test(tag)
        test = df if test is None else test.merge(
            df.drop(columns=["_submission_row"]), on=TIMESTAMP_COL, how="inner",
        )
    cf_test = np.column_stack([test[f"{tag}_pred_cf_mw"].to_numpy() for tag in MODELS])
    mw_test = np.column_stack([test[f"{tag}_pred_mw_mw"].to_numpy() for tag in MODELS])
    cf_pred_test = cf_test @ np.array(w_cf)
    mw_pred_test = mw_test @ np.array(w_mw)
    final_mw = np.clip(alpha * mw_pred_test + (1 - alpha) * cf_pred_test, 0.0, CAPACITY_MW)

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
