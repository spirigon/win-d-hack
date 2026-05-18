"""V124: Apply winning correction to v115 + blend corrected v97b & v115.

Best recipe so far: CF + hw_bias*0.7 + co_bias*0.5 + Q1_bias*0.7
  → v97b corrected = 7.352 on LB

Now:
1. Apply same recipe to v115 (AIFS+GraphCast model)
2. Blend corrected v97b + corrected v115 at various weights
3. Also try blending the raw corrected submissions directly
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL

V97B_OOF_PATH = _ROOT / "data" / "processed" / "v97b_oof.parquet"
V115_OOF_PATH = _ROOT / "data" / "processed" / "v115_oof.parquet"
V97B_SUB_CF_PATH = _ROOT / "submissions" / "archive" / "v97b.0_cfonly.csv"
V115_SUB_CF_PATH = _ROOT / "submissions" / "archive" / "v115.0_cfonly.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"


def _write(preds: np.ndarray, path: Path, label: str = "") -> None:
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<50s} mean={preds.mean():.2f}  {label}")


def main() -> None:
    print("=" * 72)
    print("V124: Blend corrected v97b + corrected v115")
    print("=" * 72)

    # --- Load OOFs ---
    oof97 = pd.read_parquet(V97B_OOF_PATH)
    oof115 = pd.read_parquet(V115_OOF_PATH)
    oof97["ts"] = pd.to_datetime(oof97["ts"])
    oof115["ts"] = pd.to_datetime(oof115["ts"])

    # --- Compute v115 CF biases ---
    oof115["error_cf"] = oof115["target_mw"] - oof115["pred_cf_mw"]
    print("\n  V115 CF error stats:")
    hw_115 = float(oof115[(oof115["ws_120"] >= 12) & (oof115["ws_120"] < 18)]["error_cf"].mean())
    co_115 = float(oof115[oof115["ws_120"] >= 18]["error_cf"].mean())
    mid_115 = float(oof115[(oof115["ws_120"] >= 8) & (oof115["ws_120"] < 12)]["error_cf"].mean())
    q1_115 = float(oof115[oof115["ts"].dt.month.isin([1, 2, 3])]["error_cf"].mean())
    global_115 = float(oof115["error_cf"].mean())
    print(f"  hw={hw_115:+.3f} co={co_115:+.3f} mid={mid_115:+.3f} q1={q1_115:+.3f} global={global_115:+.3f}")

    # V97b CF biases (from v122/v123)
    oof97["error_cf"] = oof97["target_mw"] - oof97["pred_cf_mw"]
    hw_97 = float(oof97[(oof97["ws_120"] >= 12) & (oof97["ws_120"] < 18)]["error_cf"].mean())
    co_97 = float(oof97[oof97["ws_120"] >= 18]["error_cf"].mean())
    q1_97 = float(oof97[oof97["ts"].dt.month.isin([1, 2, 3])]["error_cf"].mean())
    print(f"\n  V97b CF: hw={hw_97:+.3f} co={co_97:+.3f} q1={q1_97:+.3f}")

    # --- OOF comparison ---
    # Fold 5 nMAE for both
    for name, oof_df in [("v97b", oof97), ("v115", oof115)]:
        f5 = oof_df[oof_df["fold"] == 5]
        nmae_cf = float(np.mean(np.abs(f5["target_mw"] - f5["pred_cf_mw"])) / CAPACITY_MW * 100)
        nmae_bl = float(np.mean(np.abs(f5["target_mw"] - f5["pred_blend_mw"])) / CAPACITY_MW * 100)
        print(f"  {name} F5: CF={nmae_cf:.4f}%  blend={nmae_bl:.4f}%")

    # Correlation between v97b and v115 predictions on fold 5
    f5_97 = oof97[oof97["fold"] == 5].sort_values("ts").reset_index(drop=True)
    f5_115 = oof115[oof115["fold"] == 5].sort_values("ts").reset_index(drop=True)
    corr = float(np.corrcoef(f5_97["pred_cf_mw"].to_numpy(), f5_115["pred_cf_mw"].to_numpy())[0, 1])
    print(f"  Correlation v97b CF vs v115 CF (F5): {corr:.4f}")

    # Blend OOF: does blending the two OOF predictions improve?
    blend_preds = 0.5 * f5_97["pred_cf_mw"].to_numpy() + 0.5 * f5_115["pred_cf_mw"].to_numpy()
    blend_nmae = float(np.mean(np.abs(f5_97["target_mw"].to_numpy() - blend_preds)) / CAPACITY_MW * 100)
    print(f"  50/50 blend F5 nMAE: {blend_nmae:.4f}%")

    for w97 in [0.6, 0.7, 0.8, 0.9]:
        bp = w97 * f5_97["pred_cf_mw"].to_numpy() + (1 - w97) * f5_115["pred_cf_mw"].to_numpy()
        nm = float(np.mean(np.abs(f5_97["target_mw"].to_numpy() - bp)) / CAPACITY_MW * 100)
        print(f"  {w97:.0%} v97b / {1-w97:.0%} v115 blend F5: {nm:.4f}%")

    # --- Load test predictions ---
    v97b_cf = pd.read_csv(V97B_SUB_CF_PATH)
    v115_cf = pd.read_csv(V115_SUB_CF_PATH)
    pred_col = [c for c in v97b_cf.columns if c != TIMESTAMP_COL][0]
    test_97 = v97b_cf[pred_col].to_numpy(dtype=np.float64)
    test_115 = v115_cf[pred_col].to_numpy(dtype=np.float64)
    sub_ts = pd.to_datetime(v97b_cf[TIMESTAMP_COL])

    # Wind speed for test
    valid_df = pd.read_csv(VALID_PATH)
    valid_df[TIMESTAMP_COL] = pd.to_datetime(valid_df.iloc[:, 0])
    ws_map = dict(zip(valid_df[TIMESTAMP_COL], valid_df["wind_speed_120m"]))
    test_ws = np.array([ws_map.get(t, 8.0) for t in sub_ts], dtype=np.float32)

    hw_mask = (test_ws >= 12) & (test_ws < 18)
    co_mask = test_ws >= 18

    # --- Apply corrections ---
    print(f"\n  Generating corrected + blended submissions...")

    # Corrected v97b (winning recipe: hw*0.7 + co*0.5 + q1*0.7)
    corr_97 = test_97.copy()
    corr_97[hw_mask] += hw_97 * 0.7
    corr_97[co_mask] += co_97 * 0.5
    corr_97 += q1_97 * 0.7
    corr_97 = np.clip(corr_97, 0, CAPACITY_MW)

    # Corrected v115 (same correction recipe with v115-specific biases)
    corr_115 = test_115.copy()
    corr_115[hw_mask] += hw_115 * 0.7
    corr_115[co_mask] += co_115 * 0.5
    corr_115 += q1_115 * 0.7
    corr_115 = np.clip(corr_115, 0, CAPACITY_MW)

    # Write corrected v115 standalone
    _write(corr_115, OUTPUT_DIR / "v124.0_v115_corrected.csv", "v115+corrections")

    # Blends of corrected v97b + corrected v115
    for w97 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        blend = np.clip(w97 * corr_97 + (1 - w97) * corr_115, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v124.1_blend_{int(w97*100)}_{int((1-w97)*100)}.csv",
               f"{w97:.0%}v97b/{1-w97:.0%}v115")

    # Also try: corrected v97b + raw v115 (maybe v115 doesn't need same correction)
    for w97 in [0.7, 0.8, 0.9]:
        blend = np.clip(w97 * corr_97 + (1 - w97) * test_115, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v124.2_corr97_{int(w97*100)}_raw115_{int((1-w97)*100)}.csv",
               f"corr97 {w97:.0%} + raw115 {1-w97:.0%}")

    # Corrected v97b + corrected v115 with v97b-biases applied to v115
    # (v115 might have similar biases since same wind farm)
    corr_115_v97bias = test_115.copy()
    corr_115_v97bias[hw_mask] += hw_97 * 0.7
    corr_115_v97bias[co_mask] += co_97 * 0.5
    corr_115_v97bias += q1_97 * 0.7
    corr_115_v97bias = np.clip(corr_115_v97bias, 0, CAPACITY_MW)

    for w97 in [0.6, 0.7, 0.8]:
        blend = np.clip(w97 * corr_97 + (1 - w97) * corr_115_v97bias, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v124.3_both_v97bias_{int(w97*100)}_{int((1-w97)*100)}.csv",
               f"both use v97b biases")

    print(f"\n  Top picks:")
    print(f"    v124.0_v115_corrected.csv       (standalone v115 + its own corrections)")
    print(f"    v124.1_blend_70_30.csv          (70% corr_v97b + 30% corr_v115)")
    print(f"    v124.1_blend_80_20.csv          (80/20 — conservative diversity)")
    print(f"    v124.1_blend_90_10.csv          (90/10 — minimal v115 diversity)")


if __name__ == "__main__":
    main()
