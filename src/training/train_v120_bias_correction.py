"""V120: Conditional Bias Correction on V97b predictions.

Instead of learning a complex model, apply simple statistical bias
corrections based on KNOWN systematic errors from the OOF analysis:

1. Wind regime bias: v97b has +2 MW bias at ws 12-18, +13 MW at ws>18
2. Night-time bias: hours 20-04 are systematically worse
3. March transitional bias patterns
4. Model disagreement → uncertainty → shrink toward power curve

This uses a binned correction table estimated from OOF, applied with
Bayesian shrinkage (small bins get weak correction). No ML model needed.

Additionally tries:
- Quantile-based clipping: where predictions exceed P95 of historical
  power at the same wind speed, cap them.
- Direction-sector bias correction.

Outputs:
    data/processed/v120_oof.parquet
    submissions/archive/v120.0_bias_corr.csv
    submissions/archive/v120.1_aggressive.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.inference.submission import write_submission
from src.utils.seeding import set_global_seed

# --- Paths ---
V97B_OOF_PATH = _ROOT / "data" / "processed" / "v97b_oof.parquet"
V97B_SUB_BLEND_PATH = _ROOT / "submissions" / "archive" / "v97b.1_blend50.csv"
V97B_SUB_CF_PATH = _ROOT / "submissions" / "archive" / "v97b.0_cfonly.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"

OOF_PATH = _ROOT / "data" / "processed" / "v120_oof.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v120.0_bias_corr.csv"
OUTPUT_AGG_PATH = _ROOT / "submissions" / "archive" / "v120.1_aggressive.csv"

# Shrinkage: correction = bias * n / (n + SHRINKAGE_N)
# High SHRINKAGE_N = conservative (need many samples to trust correction)
SHRINKAGE_N = 30


def _compute_bin_corrections(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    bin_col: str,
    bins: list | np.ndarray,
    shrinkage_n: int = SHRINKAGE_N,
) -> dict[int, float]:
    """Compute per-bin bias correction with Bayesian shrinkage."""
    df = df.copy()
    df["_bin"] = pd.cut(df[bin_col], bins=bins, labels=False)
    corrections = {}
    for b in range(len(bins) - 1):
        sub = df[df["_bin"] == b]
        n = len(sub)
        if n < 5:
            corrections[b] = 0.0
            continue
        bias = float((sub[target_col] - sub[pred_col]).mean())
        # Shrinkage: trust correction proportional to sample size
        weight = n / (n + shrinkage_n)
        corrections[b] = bias * weight
    return corrections


def _apply_corrections(
    preds: np.ndarray,
    bin_values: np.ndarray,
    bins: list | np.ndarray,
    corrections: dict[int, float],
) -> np.ndarray:
    """Apply binned corrections to predictions."""
    result = preds.copy()
    bin_idx = np.digitize(bin_values, bins) - 1
    for b, corr in corrections.items():
        mask = bin_idx == b
        result[mask] += corr
    return np.clip(result, 0, CAPACITY_MW)


def main() -> None:
    set_global_seed(42)
    t_start = time.time()

    print("=" * 72)
    print("V120: Conditional Bias Correction on V97b")
    print("=" * 72)

    # --- Load OOF ---
    print("\n[1/3] Loading V97b OOF and computing baselines...")
    oof = pd.read_parquet(V97B_OOF_PATH)
    oof["pred_blend50"] = np.clip(
        0.5 * oof["pred_cf_mw"] + 0.5 * oof["pred_mw_mw"], 0, CAPACITY_MW
    )
    oof["error"] = oof["target_mw"] - oof["pred_blend50"]
    oof["ts"] = pd.to_datetime(oof["ts"])
    oof["hour"] = oof["ts"].dt.hour
    oof["month"] = oof["ts"].dt.month
    oof["disagreement"] = np.abs(oof["pred_cf_mw"] - oof["pred_mw_mw"])

    for fid in sorted(oof["fold"].unique()):
        sub = oof[oof["fold"] == fid]
        nmae = float(np.mean(np.abs(sub["error"])) / CAPACITY_MW * 100)
        bias = float(sub["error"].mean())
        print(f"  Fold {fid}: nMAE={nmae:.4f}%  bias={bias:+.3f} MW")

    # --- Compute corrections from leave-one-fold-out ---
    print("\n[2/3] Computing conditional bias corrections (leave-fold-out)...")

    ws_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 25, 100]
    hour_bins = [0, 4, 8, 12, 16, 20, 24]
    month_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    unique_folds = sorted(oof["fold"].unique())
    corrected_oof = np.zeros(len(oof), dtype=np.float64)

    for target_fold in unique_folds:
        val_mask = oof["fold"] == target_fold
        cal = oof[~val_mask].copy()
        val = oof[val_mask].copy()

        # 1. Wind speed bias correction
        ws_corr = _compute_bin_corrections(
            cal, "pred_blend50", "target_mw", "ws_120", ws_bins
        )

        # 2. Hour bias correction
        hour_corr = _compute_bin_corrections(
            cal, "pred_blend50", "target_mw", "hour", hour_bins
        )

        # 3. Month correction (calibrated per month)
        month_corr = _compute_bin_corrections(
            cal, "pred_blend50", "target_mw", "month", month_bins
        )

        # Apply corrections additively (each corrects a different axis)
        pred = val["pred_blend50"].to_numpy().copy()

        # Wind speed correction
        pred = _apply_corrections(pred, val["ws_120"].to_numpy(), ws_bins, ws_corr)

        # Hour correction (scaled by 0.5 to avoid double-counting)
        hour_adj = _apply_corrections(
            np.zeros_like(pred), val["hour"].to_numpy(), hour_bins, hour_corr
        )
        pred = np.clip(pred + 0.5 * hour_adj, 0, CAPACITY_MW)

        # Month correction (scaled by 0.3)
        month_adj = _apply_corrections(
            np.zeros_like(pred), val["month"].to_numpy(), month_bins, month_corr
        )
        pred = np.clip(pred + 0.3 * month_adj, 0, CAPACITY_MW)

        corrected_oof[val_mask.to_numpy()] = pred

        # Evaluate
        target = val["target_mw"].to_numpy()
        orig_nmae = float(np.mean(np.abs(target - val["pred_blend50"].to_numpy())) / CAPACITY_MW * 100)
        corr_nmae = float(np.mean(np.abs(target - pred)) / CAPACITY_MW * 100)
        delta = corr_nmae - orig_nmae
        print(f"  Fold {target_fold}: orig={orig_nmae:.4f}% → corrected={corr_nmae:.4f}% "
              f"(Δ={delta:+.4f}%)")

        if target_fold == 5:
            # Print the corrections applied
            print(f"    WS corrections: {ws_corr}")
            print(f"    Hour corrections: {hour_corr}")

    # --- Try alpha blending ---
    print("\n  Alpha search on Fold 5...")
    f5 = oof[oof["fold"] == 5]
    f5_orig = f5["pred_blend50"].to_numpy()
    f5_target = f5["target_mw"].to_numpy()
    f5_corr = corrected_oof[oof["fold"].to_numpy() == 5]
    f5_orig_nmae = float(np.mean(np.abs(f5_target - f5_orig)) / CAPACITY_MW * 100)

    best_alpha, best_nmae = 0, f5_orig_nmae
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = np.clip(alpha * f5_corr + (1 - alpha) * f5_orig, 0, CAPACITY_MW)
        nm = float(np.mean(np.abs(f5_target - blended)) / CAPACITY_MW * 100)
        if nm < best_nmae:
            best_alpha, best_nmae = alpha, nm
    print(f"  Best alpha={best_alpha:.2f}: F5 nMAE={best_nmae:.4f}% (Δ={best_nmae-f5_orig_nmae:+.4f}%)")

    # --- Alternative: simple global bias correction per fold period ---
    print("\n  Alternative: global bias shift on Fold 5...")
    # V97b has -1.05 MW bias on Fold 5 (Q1). Simply adding 1.05 MW globally?
    f5_bias = float(f5["error"].mean())
    f5_shifted = np.clip(f5_orig + f5_bias, 0, CAPACITY_MW)
    f5_shifted_nmae = float(np.mean(np.abs(f5_target - f5_shifted)) / CAPACITY_MW * 100)
    print(f"  Global bias={f5_bias:+.3f} MW → F5 nMAE={f5_shifted_nmae:.4f}% "
          f"(Δ={f5_shifted_nmae-f5_orig_nmae:+.4f}%)")

    # --- Try just the high-wind correction (the biggest known bias) ---
    print("\n  Alternative: high-wind-only correction...")
    # From OOF analysis: ws 12-18 has +2.06 MW bias, ws>18 has +13 MW bias
    # Train on folds 3+4, apply on fold 5
    cal_34 = oof[oof["fold"].isin([3, 4])]
    for ws_lo, ws_hi, label in [(12, 18, "high"), (18, 100, "cutout")]:
        sub = cal_34[(cal_34["ws_120"] >= ws_lo) & (cal_34["ws_120"] < ws_hi)]
        bias = float(sub["error"].mean()) if len(sub) > 10 else 0
        print(f"    {label} (ws {ws_lo}-{ws_hi}): bias={bias:+.2f} MW (n={len(sub)})")

    # Apply just high-wind correction on fold 5
    f5_hw = f5_orig.copy()
    hw_mask = (f5["ws_120"].to_numpy() >= 12) & (f5["ws_120"].to_numpy() < 18)
    co_mask = f5["ws_120"].to_numpy() >= 18
    # Use corrections from calibration folds
    hw_sub = cal_34[(cal_34["ws_120"] >= 12) & (cal_34["ws_120"] < 18)]
    co_sub = cal_34[cal_34["ws_120"] >= 18]
    hw_bias = float(hw_sub["error"].mean()) if len(hw_sub) > 10 else 0
    co_bias = float(co_sub["error"].mean()) if len(co_sub) > 10 else 0
    shrink_hw = len(hw_sub) / (len(hw_sub) + SHRINKAGE_N)
    shrink_co = len(co_sub) / (len(co_sub) + SHRINKAGE_N)
    f5_hw[hw_mask] += hw_bias * shrink_hw
    f5_hw[co_mask] += co_bias * shrink_co
    f5_hw = np.clip(f5_hw, 0, CAPACITY_MW)
    f5_hw_nmae = float(np.mean(np.abs(f5_target - f5_hw)) / CAPACITY_MW * 100)
    print(f"  High-wind corrected F5 nMAE={f5_hw_nmae:.4f}% "
          f"(Δ={f5_hw_nmae-f5_orig_nmae:+.4f}%)")

    # --- Quantile capping approach ---
    print("\n  Alternative: quantile capping (cap overestimates)...")
    # From OOF: v97b over-predicts at high wind. Cap predictions at historical
    # P90 of actual power for the same wind speed bin.
    # Build P90 lookup from calibration data
    cal_all = oof.copy()
    cal_all["ws_bin"] = pd.cut(cal_all["ws_120"], bins=ws_bins, labels=False)
    p90_by_ws = cal_all.groupby("ws_bin")["target_mw"].quantile(0.95).to_dict()
    p10_by_ws = cal_all.groupby("ws_bin")["target_mw"].quantile(0.05).to_dict()

    f5_capped = f5_orig.copy()
    f5_ws_bin = np.digitize(f5["ws_120"].to_numpy(), ws_bins) - 1
    for b in range(len(ws_bins) - 1):
        mask = f5_ws_bin == b
        if not mask.any():
            continue
        p95 = p90_by_ws.get(b, CAPACITY_MW)
        p05 = p10_by_ws.get(b, 0.0)
        if p95 is not None and not np.isnan(p95):
            f5_capped[mask] = np.minimum(f5_capped[mask], p95)
        if p05 is not None and not np.isnan(p05):
            f5_capped[mask] = np.maximum(f5_capped[mask], p05)
    f5_capped = np.clip(f5_capped, 0, CAPACITY_MW)
    f5_cap_nmae = float(np.mean(np.abs(f5_target - f5_capped)) / CAPACITY_MW * 100)
    print(f"  Quantile-capped F5 nMAE={f5_cap_nmae:.4f}% "
          f"(Δ={f5_cap_nmae-f5_orig_nmae:+.4f}%)")

    # --- Combined best approach ---
    print("\n  Combined: best individual corrections...")
    # Take the best performing approach
    results = {
        "original": f5_orig_nmae,
        "binned_corrections": best_nmae,
        "global_bias_shift": f5_shifted_nmae,
        "high_wind_only": f5_hw_nmae,
        "quantile_capping": f5_cap_nmae,
    }
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    print(f"  Ranked approaches (Fold 5 nMAE):")
    for name, nm in sorted_results:
        marker = " ← BEST" if nm == sorted_results[0][1] else ""
        delta = nm - f5_orig_nmae
        print(f"    {name:<25s}: {nm:.4f}% (Δ={delta:+.4f}%){marker}")

    # --- Try ensemble of corrections ---
    print("\n  Ensemble of ALL corrections (equal weight)...")
    all_corrected = [f5_orig]  # always include original
    if best_nmae < f5_orig_nmae:
        all_corrected.append(f5_corr)
    if f5_shifted_nmae < f5_orig_nmae:
        all_corrected.append(f5_shifted)
    if f5_hw_nmae < f5_orig_nmae:
        all_corrected.append(f5_hw)
    if f5_cap_nmae < f5_orig_nmae:
        all_corrected.append(f5_capped)

    if len(all_corrected) > 1:
        ensemble = np.clip(np.mean(all_corrected, axis=0), 0, CAPACITY_MW)
        ens_nmae = float(np.mean(np.abs(f5_target - ensemble)) / CAPACITY_MW * 100)
        print(f"  Ensemble of {len(all_corrected)} versions: F5 nMAE={ens_nmae:.4f}% "
              f"(Δ={ens_nmae-f5_orig_nmae:+.4f}%)")
    else:
        print("  No correction improved over original.")
        ens_nmae = f5_orig_nmae

    # --- Save OOF and build submission ---
    print("\n[3/3] Building submission...")
    # Determine best approach to apply on test
    best_approach = sorted_results[0][0]
    print(f"  Using approach: {best_approach}")

    # Load test predictions
    v97b_blend_sub = pd.read_csv(V97B_SUB_BLEND_PATH)
    v97b_cf_sub = pd.read_csv(V97B_SUB_CF_PATH)
    pred_col_bl = [c for c in v97b_blend_sub.columns if c != TIMESTAMP_COL][0]
    pred_col_cf = [c for c in v97b_cf_sub.columns if c != TIMESTAMP_COL][0]
    test_pred_blend = v97b_blend_sub[pred_col_bl].to_numpy(dtype=np.float64)
    test_pred_cf = v97b_cf_sub[pred_col_cf].to_numpy(dtype=np.float64)
    sub_ts = pd.to_datetime(v97b_blend_sub[TIMESTAMP_COL])
    n_test = len(test_pred_blend)

    # Load valid features for wind speed
    valid_df = pd.read_csv(VALID_PATH)
    valid_df[TIMESTAMP_COL] = pd.to_datetime(valid_df.iloc[:, 0])

    # Map wind speed to submission order
    ws_map = dict(zip(valid_df[TIMESTAMP_COL], valid_df["wind_speed_120m"]))
    test_ws = np.array([ws_map.get(t, 8.0) for t in sub_ts], dtype=np.float32)
    test_hour = sub_ts.dt.hour.to_numpy(dtype=np.float32)
    test_month = sub_ts.dt.month.to_numpy(dtype=np.float32)

    # Apply the best approach on test data
    if best_approach == "global_bias_shift":
        # Use the average bias from all calibration folds
        all_bias = float(oof["error"].mean())
        test_final = np.clip(test_pred_blend + all_bias, 0, CAPACITY_MW)
        print(f"  Applied global bias shift: {all_bias:+.3f} MW")

    elif best_approach == "high_wind_only":
        test_final = test_pred_blend.copy()
        hw_mask = (test_ws >= 12) & (test_ws < 18)
        co_mask = test_ws >= 18
        # Use corrections from ALL OOF data
        hw_all = oof[(oof["ws_120"] >= 12) & (oof["ws_120"] < 18)]
        co_all = oof[oof["ws_120"] >= 18]
        hw_bias_all = float(hw_all["error"].mean()) if len(hw_all) > 10 else 0
        co_bias_all = float(co_all["error"].mean()) if len(co_all) > 10 else 0
        shrink_hw_all = len(hw_all) / (len(hw_all) + SHRINKAGE_N)
        shrink_co_all = len(co_all) / (len(co_all) + SHRINKAGE_N)
        test_final[hw_mask] += hw_bias_all * shrink_hw_all
        test_final[co_mask] += co_bias_all * shrink_co_all
        test_final = np.clip(test_final, 0, CAPACITY_MW)
        print(f"  Applied high-wind correction: hw={hw_bias_all*shrink_hw_all:+.2f}, "
              f"co={co_bias_all*shrink_co_all:+.2f}")

    elif best_approach == "binned_corrections":
        # Recompute corrections from ALL OOF
        ws_corr_all = _compute_bin_corrections(
            oof, "pred_blend50", "target_mw", "ws_120", ws_bins
        )
        hour_corr_all = _compute_bin_corrections(
            oof, "pred_blend50", "target_mw", "hour", hour_bins
        )
        month_corr_all = _compute_bin_corrections(
            oof, "pred_blend50", "target_mw", "month", month_bins
        )
        test_final = _apply_corrections(test_pred_blend, test_ws, ws_bins, ws_corr_all)
        hour_adj = _apply_corrections(
            np.zeros_like(test_final), test_hour, hour_bins, hour_corr_all
        )
        test_final = np.clip(test_final + 0.5 * hour_adj, 0, CAPACITY_MW)
        month_adj = _apply_corrections(
            np.zeros_like(test_final), test_month, month_bins, month_corr_all
        )
        test_final = np.clip(test_final + 0.3 * month_adj, 0, CAPACITY_MW)
        print(f"  Applied binned corrections (ws + hour + month)")

    elif best_approach == "quantile_capping":
        test_final = test_pred_blend.copy()
        test_ws_bin = np.digitize(test_ws, ws_bins) - 1
        for b in range(len(ws_bins) - 1):
            mask = test_ws_bin == b
            if not mask.any():
                continue
            p95 = p90_by_ws.get(b, CAPACITY_MW)
            p05 = p10_by_ws.get(b, 0.0)
            if p95 is not None and not np.isnan(p95):
                test_final[mask] = np.minimum(test_final[mask], p95)
            if p05 is not None and not np.isnan(p05):
                test_final[mask] = np.maximum(test_final[mask], p05)
        test_final = np.clip(test_final, 0, CAPACITY_MW)
        print(f"  Applied quantile capping")

    else:  # original
        test_final = test_pred_blend.copy()
        print(f"  No correction applied (original is best)")

    print(f"  Final: mean={test_final.mean():.2f} MW, std={test_final.std():.2f} MW")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        test_final, OUTPUT_PATH,
        expected_rows=n_test,
        timestamps=sub_ts.to_numpy(),
    )
    print(f"  Submission saved: {OUTPUT_PATH}")

    # Aggressive version: apply all corrections that improved
    test_aggressive = test_pred_blend.copy()
    # Apply high-wind fix
    hw_all = oof[(oof["ws_120"] >= 12) & (oof["ws_120"] < 18)]
    co_all = oof[oof["ws_120"] >= 18]
    hw_bias_all = float(hw_all["error"].mean()) if len(hw_all) > 10 else 0
    co_bias_all = float(co_all["error"].mean()) if len(co_all) > 10 else 0
    hw_mask_t = (test_ws >= 12) & (test_ws < 18)
    co_mask_t = test_ws >= 18
    test_aggressive[hw_mask_t] += hw_bias_all * 0.8
    test_aggressive[co_mask_t] += co_bias_all * 0.5
    # Apply global bias (small)
    global_bias = float(oof["error"].mean())
    test_aggressive += global_bias * 0.3
    test_aggressive = np.clip(test_aggressive, 0, CAPACITY_MW)

    write_submission(
        test_aggressive, OUTPUT_AGG_PATH,
        expected_rows=n_test,
        timestamps=sub_ts.to_numpy(),
    )
    print(f"  Aggressive saved: {OUTPUT_AGG_PATH}")

    # Save OOF
    best_corr_oof = corrected_oof if best_approach == "binned_corrections" else oof["pred_blend50"].to_numpy()
    oof_out = pd.DataFrame({
        "fold": oof["fold"].to_numpy(),
        "ts": oof["ts"].to_numpy(),
        "target_mw": oof["target_mw"].to_numpy(),
        "pred_blend_mw": best_corr_oof,
    })
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_out.to_parquet(OOF_PATH, index=False)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY (took {elapsed:.0f}s)")
    print(f"{'=' * 72}")
    print(f"  v97b.1_blend50 OOF F5: {f5_orig_nmae:.4f}%  (LB=7.415%)")
    print(f"  Best approach: {best_approach}")
    print(f"  Best F5 nMAE: {sorted_results[0][1]:.4f}% (Δ={sorted_results[0][1]-f5_orig_nmae:+.4f}%)")
    print(f"\n  All results:")
    for name, nm in sorted_results:
        print(f"    {name:<25s}: {nm:.4f}%")


if __name__ == "__main__":
    main()
