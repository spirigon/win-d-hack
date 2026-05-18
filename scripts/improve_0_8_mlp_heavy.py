"""MLP-heavy blend for hours 0-8.

The full-data MLP showed 2.5% nMAE on spring-2025 validation — the best
individual model. Currently it's 10% of the blend. For the scored hours
0-8, give it 50% weight.

Blend: 50% MLP + 25% LGBM_CF + 25% LGBM_MW
(TFT excluded because its predictions are already embedded in the
existing blend — this script re-blends from raw model outputs.)

Then apply persistence correction for hours 0-2 (farm was off at 23:00).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.data.schema import CAPACITY_MW

OUTPUT = _ROOT / "submissions" / "18_05_2026_mlp_heavy_0_8.csv"
EXISTING = _ROOT / "submissions" / "18_05_2026_forecast.csv"

# From the hybrid run output (predict_spring.txt / latest run):
# MLP predictions for each hour (from the FULL DATA trained MLP, hybrid run)
MLP_PREDS = {
    0: 0.699, 1: 1.368, 2: 0.802, 3: 1.071,
    4: 5.701, 5: 10.246, 6: 14.007, 7: 9.699, 8: 7.382,
}
# LGBM CF predictions
LGBM_CF = {
    0: 1.571, 1: 1.616, 2: 1.323, 3: 2.543,
    4: 7.988, 5: 9.321, 6: 5.270, 7: 4.059, 8: 6.195,
}
# LGBM MW predictions
LGBM_MW = {
    0: 1.402, 1: 1.517, 2: 1.338, 3: 2.494,
    4: 8.022, 5: 8.796, 6: 4.949, 7: 3.863, 8: 5.377,
}
# TFT P50 — from the spring-only TFT (best sequence model)
# We don't have the per-hour TFT values directly but can infer from the blend.
# existing_blend = 0.30*CF + 0.30*MW + 0.10*MLP_spring + 0.30*TFT
# Solving for TFT: TFT = (blend - 0.30*CF - 0.30*MW - 0.10*MLP_spring) / 0.30
# Using the spring run output:
EXISTING_BLEND_SPRING = {
    0: 1.933, 1: 2.043, 2: 1.497, 3: 2.916,
    4: 7.370, 5: 8.658, 6: 5.018, 7: 4.048, 8: 5.758,
}
MLP_SPRING = {  # from the spring-only MLP (10.7% nMAE)
    0: 1.766, 1: 2.285, 2: 0.962, 3: 3.387,
    4: 9.112, 5: 12.594, 6: 5.188, 7: 5.089, 8: 6.433,
}

MAY17_23_POWER = 0.076  # farm was OFF


def main():
    # Derive TFT predictions from existing blend
    tft = {}
    for h in range(9):
        # blend = 0.30*CF + 0.30*MW + 0.10*MLP_spring + 0.30*TFT
        tft_val = (EXISTING_BLEND_SPRING[h] - 0.30*LGBM_CF[h] - 0.30*LGBM_MW[h] - 0.10*MLP_SPRING[h]) / 0.30
        tft[h] = max(0, tft_val)

    print("Derived TFT P50 predictions:")
    for h in range(9):
        print(f"  h{h}: TFT={tft[h]:.2f}")

    # Now blend with HEAVY MLP weight:
    # Strategy: 40% MLP(full) + 20% LGBM_CF + 20% LGBM_MW + 20% TFT(spring)
    W_MLP = 0.40
    W_CF  = 0.20
    W_MW  = 0.20
    W_TFT = 0.20

    print(f"\nBlend weights: MLP={W_MLP} CF={W_CF} MW={W_MW} TFT={W_TFT}")
    print(f"\n{'Hour':>5}  {'MLP':>6}  {'CF':>6}  {'MW':>6}  {'TFT':>6}  {'Blend':>7}  {'Final':>7}")
    print(f"  {'-'*50}")

    final = {}
    for h in range(9):
        blend = W_MLP * MLP_PREDS[h] + W_CF * LGBM_CF[h] + W_MW * LGBM_MW[h] + W_TFT * tft[h]
        blend = max(0, min(blend, CAPACITY_MW))

        # Persistence correction for hours 0-2
        if h <= 2:
            # 50% blend + 50% persistence from 23:00 (0.076 MW)
            final_val = 0.5 * blend + 0.5 * MAY17_23_POWER
        elif h == 3:
            # 70% blend + 30% persistence
            final_val = 0.7 * blend + 0.3 * MAY17_23_POWER
        else:
            final_val = blend

        final[h] = max(0, min(final_val, CAPACITY_MW))
        print(f"  {h:>2}h   {MLP_PREDS[h]:>5.2f}  {LGBM_CF[h]:>5.2f}  "
              f"{LGBM_MW[h]:>5.2f}  {tft[h]:>5.2f}  {blend:>6.2f}  {final[h]:>6.2f}")

    mean_final = np.mean(list(final.values()))
    print(f"\n  Mean (hours 0-8): {mean_final:.3f} MW")

    # Build full 24-hour submission
    existing = pd.read_csv(EXISTING)
    existing["datetime"] = pd.to_datetime(existing["datetime"])
    out = existing[["datetime", "hour", "forecast_mw"]].copy()

    # Replace hours 0-8
    for h, v in final.items():
        out.loc[out["hour"] == h, "forecast_mw"] = v

    out["forecast_mw"] = out["forecast_mw"].clip(0, CAPACITY_MW)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"\n  Full 24h mean: {out['forecast_mw'].mean():.3f} MW")
    print(f"  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
