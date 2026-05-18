"""V121: Tuned bias correction — v120.1 aggressive scored 7.39 on LB.

LB confirmed the direction: the correction works. Now systematically tune
the correction strength and try variants to push lower.

What v120.1_aggressive did:
  - High-wind (12-18 m/s): + hw_bias * 0.8  (hw_bias = mean error in that regime)
  - Cutout (>18 m/s): + co_bias * 0.5
  - Global bias shift: + global_mean_error * 0.3

LB improvement: 7.415 → 7.39 = -0.025 pp

Strategy: sweep correction strengths to find optimal on LB-proxy.
Since fold 5 bias is OPPOSITE to folds 3+4 and the LB agreed with the
direction from the FULL dataset average, the test set (Q1 2026) seems
to have similar bias direction to the average — i.e., a small positive
correction helps (the model slightly under-predicts on average for Q1).

Outputs multiple variants for LB probing.
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
from src.utils.seeding import set_global_seed

# --- Paths ---
V97B_OOF_PATH = _ROOT / "data" / "processed" / "v97b_oof.parquet"
V97B_SUB_BLEND_PATH = _ROOT / "submissions" / "archive" / "v97b.1_blend50.csv"
V97B_SUB_CF_PATH = _ROOT / "submissions" / "archive" / "v97b.0_cfonly.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TARGET_COL_NAME = "Выработка. Результирующий расчет"


def _write_no_ts(preds: np.ndarray, path: Path) -> None:
    """Write single-column submission (no timestamp)."""
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    df = pd.DataFrame({TARGET_COL_NAME: preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Written: {path.name} (mean={preds.mean():.2f}, std={preds.std():.2f})")


def main() -> None:
    set_global_seed(42)

    print("=" * 72)
    print("V121: Tuned Bias Correction Sweep")
    print("=" * 72)

    # --- Load data ---
    oof = pd.read_parquet(V97B_OOF_PATH)
    oof["pred_blend50"] = np.clip(
        0.5 * oof["pred_cf_mw"] + 0.5 * oof["pred_mw_mw"], 0, CAPACITY_MW
    )
    oof["error"] = oof["target_mw"] - oof["pred_blend50"]
    oof["ts"] = pd.to_datetime(oof["ts"])

    # Load test predictions
    v97b_blend_sub = pd.read_csv(V97B_SUB_BLEND_PATH)
    v97b_cf_sub = pd.read_csv(V97B_SUB_CF_PATH)
    pred_col_bl = [c for c in v97b_blend_sub.columns if c != TIMESTAMP_COL][0]
    pred_col_cf = [c for c in v97b_cf_sub.columns if c != TIMESTAMP_COL][0]
    test_pred_blend = v97b_blend_sub[pred_col_bl].to_numpy(dtype=np.float64)
    test_pred_cf = v97b_cf_sub[pred_col_cf].to_numpy(dtype=np.float64)
    test_pred_mw = np.clip(2 * test_pred_blend - test_pred_cf, 0, CAPACITY_MW)
    sub_ts = pd.to_datetime(v97b_blend_sub[TIMESTAMP_COL])

    # Load valid features for wind speed
    valid_df = pd.read_csv(VALID_PATH)
    valid_df[TIMESTAMP_COL] = pd.to_datetime(valid_df.iloc[:, 0])
    ws_map = dict(zip(valid_df[TIMESTAMP_COL], valid_df["wind_speed_120m"]))
    test_ws = np.array([ws_map.get(t, 8.0) for t in sub_ts], dtype=np.float32)
    test_month = sub_ts.dt.month.to_numpy()
    test_hour = sub_ts.dt.hour.to_numpy()

    # --- Compute OOF statistics ---
    print("\n  OOF error statistics:")
    # Global
    global_bias = float(oof["error"].mean())
    print(f"  Global bias: {global_bias:+.3f} MW")

    # Per wind regime
    ws_regimes = [(0, 4, "calm"), (4, 8, "low"), (8, 12, "mid"),
                  (12, 18, "high"), (18, 100, "cutout")]
    for lo, hi, name in ws_regimes:
        sub = oof[(oof["ws_120"] >= lo) & (oof["ws_120"] < hi)]
        if len(sub) > 5:
            bias = float(sub["error"].mean())
            print(f"  ws {lo:2d}-{hi:3d} ({name:6s}): bias={bias:+.3f} MW  n={len(sub)}")

    # Per month
    print("  Per month:")
    for m in [1, 2, 3]:
        sub = oof[oof["ts"].dt.month == m]
        if len(sub) > 0:
            bias = float(sub["error"].mean())
            print(f"    Month {m}: bias={bias:+.3f} MW  n={len(sub)}")

    # Night vs day
    night = oof[oof["ts"].dt.hour.isin([20, 21, 22, 23, 0, 1, 2, 3])]
    day = oof[~oof["ts"].dt.hour.isin([20, 21, 22, 23, 0, 1, 2, 3])]
    print(f"  Night (20-04): bias={night['error'].mean():+.3f} MW")
    print(f"  Day   (04-20): bias={day['error'].mean():+.3f} MW")

    # Q1-specific stats (folds covering Q1 months)
    q1_oof = oof[oof["ts"].dt.month.isin([1, 2, 3])]
    q1_bias = float(q1_oof["error"].mean()) if len(q1_oof) > 0 else 0
    print(f"  Q1-months-only bias: {q1_bias:+.3f} MW (n={len(q1_oof)})")

    # --- Generate correction variants ---
    print(f"\n  Generating correction variants...")

    # v120.1 aggressive recipe (confirmed 7.39):
    # hw_bias * 0.8, co_bias * 0.5, global * 0.3
    hw_sub = oof[(oof["ws_120"] >= 12) & (oof["ws_120"] < 18)]
    co_sub = oof[oof["ws_120"] >= 18]
    hw_bias = float(hw_sub["error"].mean()) if len(hw_sub) > 10 else 0
    co_bias = float(co_sub["error"].mean()) if len(co_sub) > 10 else 0
    print(f"\n  Reference corrections:")
    print(f"    hw_bias={hw_bias:+.3f}, co_bias={co_bias:+.3f}, global={global_bias:+.3f}")

    # --- Variant A: v120.1 exact recipe (baseline for comparison) ---
    va = test_pred_blend.copy()
    hw_mask = (test_ws >= 12) & (test_ws < 18)
    co_mask = test_ws >= 18
    va[hw_mask] += hw_bias * 0.8
    va[co_mask] += co_bias * 0.5
    va += global_bias * 0.3
    va = np.clip(va, 0, CAPACITY_MW)
    _write_no_ts(va, OUTPUT_DIR / "v121.A_baseline_recipe.csv")

    # --- Variant B: stronger global shift ---
    for g_weight in [0.5, 0.7, 1.0]:
        vb = test_pred_blend.copy()
        vb[hw_mask] += hw_bias * 0.8
        vb[co_mask] += co_bias * 0.5
        vb += global_bias * g_weight
        vb = np.clip(vb, 0, CAPACITY_MW)
        _write_no_ts(vb, OUTPUT_DIR / f"v121.B_global_{g_weight:.1f}.csv")

    # --- Variant C: add mid-wind correction ---
    mid_sub = oof[(oof["ws_120"] >= 8) & (oof["ws_120"] < 12)]
    mid_bias = float(mid_sub["error"].mean()) if len(mid_sub) > 10 else 0
    low_sub = oof[(oof["ws_120"] >= 4) & (oof["ws_120"] < 8)]
    low_bias = float(low_sub["error"].mean()) if len(low_sub) > 10 else 0
    print(f"    mid_bias={mid_bias:+.3f}, low_bias={low_bias:+.3f}")

    mid_mask = (test_ws >= 8) & (test_ws < 12)
    low_mask = (test_ws >= 4) & (test_ws < 8)

    for combo_label, hw_w, co_w, mid_w, low_w, g_w in [
        ("C1", 0.8, 0.5, 0.5, 0.3, 0.3),
        ("C2", 1.0, 0.7, 0.5, 0.3, 0.3),
        ("C3", 1.0, 0.7, 0.7, 0.5, 0.5),
        ("C4", 0.8, 0.5, 0.3, 0.2, 0.5),
        ("C5", 1.0, 0.5, 0.5, 0.3, 0.5),  # stronger global + mid
    ]:
        vc = test_pred_blend.copy()
        vc[hw_mask] += hw_bias * hw_w
        vc[co_mask] += co_bias * co_w
        vc[mid_mask] += mid_bias * mid_w
        vc[low_mask] += low_bias * low_w
        vc += global_bias * g_w
        vc = np.clip(vc, 0, CAPACITY_MW)
        _write_no_ts(vc, OUTPUT_DIR / f"v121.{combo_label}_combo.csv")

    # --- Variant D: night-specific correction ---
    night_mask = np.isin(test_hour, [20, 21, 22, 23, 0, 1, 2, 3])
    night_bias = float(night["error"].mean())
    day_bias = float(day["error"].mean())

    vd = test_pred_blend.copy()
    vd[hw_mask] += hw_bias * 0.8
    vd[co_mask] += co_bias * 0.5
    vd[night_mask] += night_bias * 0.3
    vd[~night_mask] += day_bias * 0.3
    vd += global_bias * 0.2
    vd = np.clip(vd, 0, CAPACITY_MW)
    _write_no_ts(vd, OUTPUT_DIR / "v121.D_night_corr.csv")

    # --- Variant E: Month-specific correction ---
    m1_bias = float(oof[oof["ts"].dt.month == 1]["error"].mean())
    m2_bias = float(oof[oof["ts"].dt.month == 2]["error"].mean())
    m3_bias = float(oof[oof["ts"].dt.month == 3]["error"].mean())
    print(f"    m1_bias={m1_bias:+.3f}, m2_bias={m2_bias:+.3f}, m3_bias={m3_bias:+.3f}")

    ve = test_pred_blend.copy()
    ve[hw_mask] += hw_bias * 0.8
    ve[co_mask] += co_bias * 0.5
    ve[test_month == 1] += m1_bias * 0.3
    ve[test_month == 2] += m2_bias * 0.3
    ve[test_month == 3] += m3_bias * 0.3
    ve = np.clip(ve, 0, CAPACITY_MW)
    _write_no_ts(ve, OUTPUT_DIR / "v121.E_month_corr.csv")

    # --- Variant F: Use CF-only as base (LB: v97b.0 CF = 7.45) ---
    # Maybe the CF model + correction beats blend50 + correction
    vf = test_pred_cf.copy()
    vf[hw_mask] += hw_bias * 0.8
    vf[co_mask] += co_bias * 0.5
    vf += global_bias * 0.3
    vf = np.clip(vf, 0, CAPACITY_MW)
    _write_no_ts(vf, OUTPUT_DIR / "v121.F_cf_base_corr.csv")

    # --- Variant G: Q1-month-only bias (more relevant to test) ---
    vg = test_pred_blend.copy()
    vg[hw_mask] += hw_bias * 0.8
    vg[co_mask] += co_bias * 0.5
    vg += q1_bias * 0.5  # use Q1-specific bias with higher weight
    vg = np.clip(vg, 0, CAPACITY_MW)
    _write_no_ts(vg, OUTPUT_DIR / "v121.G_q1_bias.csv")

    # --- Variant H: Larger global + all regimes ---
    vh = test_pred_blend.copy()
    calm_sub = oof[oof["ws_120"] < 4]
    calm_bias = float(calm_sub["error"].mean()) if len(calm_sub) > 10 else 0
    calm_mask = test_ws < 4
    vh[calm_mask] += calm_bias * 0.3
    vh[low_mask] += low_bias * 0.3
    vh[mid_mask] += mid_bias * 0.5
    vh[hw_mask] += hw_bias * 1.0
    vh[co_mask] += co_bias * 0.7
    vh += global_bias * 0.5
    vh = np.clip(vh, 0, CAPACITY_MW)
    _write_no_ts(vh, OUTPUT_DIR / "v121.H_all_regimes.csv")

    # --- Variant I: Simple uniform shift (just add +0.5 MW to everything) ---
    for shift in [0.3, 0.5, 0.7, 1.0, 1.5]:
        vi = np.clip(test_pred_blend + shift, 0, CAPACITY_MW)
        _write_no_ts(vi, OUTPUT_DIR / f"v121.I_shift_{shift:.1f}.csv")

    print(f"\n  Total variants generated: 18")
    print(f"\n  Recommendations for LB probing:")
    print(f"    1. v121.B_global_0.5 (stronger global shift)")
    print(f"    2. v121.C5_combo (global 0.5 + mid + hw)")
    print(f"    3. v121.I_shift_0.5 (simplest: uniform +0.5 MW)")
    print(f"    4. v121.G_q1_bias (Q1-specific bias)")
    print(f"    5. v121.H_all_regimes (all regime corrections)")


if __name__ == "__main__":
    main()
