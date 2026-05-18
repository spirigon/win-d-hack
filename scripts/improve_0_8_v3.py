"""V3: Use May 17 actuals + data-driven calibration for hours 0-8.

Key data:
  May 17 21:00 actual = 6.84 MW (brief gust, NWP says ws=1.0)
  May 17 22:00 actual = 2.58 MW
  May 17 23:00 actual = 0.076 MW (farm OFF going into midnight)

Persistence rule: if the farm was OFF at 23:00 and NWP says wind stays
sub-3 m/s for the next 3 hours, it's extremely likely the farm stays
off through hours 0-2.

The ramp at hours 3-5 (NWP ws 3.2 → 4.75) is uncertain. The evening
showed the NWP is unreliable (predicted 1.0 m/s when actual production
was 6.8 MW). So we hedge: use a blend of the model prediction and the
conditional median, but anchor hours 0-2 to the persistence of zero.

Final approach per hour:
  h0-h2: persistence from 23:00 (0.076 MW) → predict ~0.3-0.5 MW
         (allow a small buffer for measurement noise + restart delay)
  h3:    transition — NWP says 3.2 m/s (at cut-in). Model says 2.68.
         Conditional median at 3.0-3.5 in spring = 1.80. Blend: ~2.0 MW.
  h4-h8: the model + conditional median blend (v2 approach).
         BUT the May 17 evening showed the NWP systematically
         under-forecasts. If actual ws at hour 4 is higher than NWP,
         the model is already under-predicting. Use the v2 calibration
         which pulls toward the conditional median (higher than model).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW

INPUT   = _ROOT / "submissions" / "18_05_2026_forecast.csv"
OUTPUT  = _ROOT / "submissions" / "18_05_2026_final_0_8_v3.csv"

# May 17 evening actuals (from the test file)
MAY17_LAST_ACTUAL = 0.076   # MW at 23:00 — farm was OFF

# NWP data for May 18
MAY18_WS120 = {
    0: 1.34, 1: 2.52, 2: 2.02, 3: 3.20,
    4: 4.75, 5: 4.60, 6: 4.40, 7: 3.75, 8: 4.53,
}

# Conditional medians from historical Apr-May hours 0-8 data
# (computed in improve_0_8_v2.py from the training set)
COND_MEDIANS = {
    # ws_bin: median_power_MW
    (1.0, 1.5): 2.13,
    (1.5, 2.0): 0.70,
    (2.0, 2.5): 1.68,
    (2.5, 3.0): 2.48,
    (3.0, 3.5): 1.80,
    (3.5, 4.0): 4.87,
    (4.0, 4.5): 6.14,
    (4.5, 5.0): 11.31,
}


def get_cond_median(ws: float) -> float:
    for (lo, hi), med in COND_MEDIANS.items():
        if lo <= ws < hi:
            return med
    return 0.0


def main():
    df = pd.read_csv(INPUT)
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = df["hour"].between(0, 8)
    out = df[mask].copy().sort_values("hour").reset_index(drop=True)

    model_pred = out["forecast_mw"].to_numpy().copy()
    final = np.zeros(9)

    print("V3 prediction (persistence + data-calibrated blend):")
    print(f"  May 17 23:00 actual: {MAY17_LAST_ACTUAL:.3f} MW (farm OFF)")
    print()
    print(f"{'Hour':>5}  {'NWP ws':>7}  {'Model':>7}  {'CondMed':>8}  {'Final':>7}  {'Logic'}")
    print(f"  {'-'*65}")

    for h in range(9):
        ws = MAY18_WS120[h]
        mp = model_pred[h]
        cm = get_cond_median(ws)

        if h <= 2:
            # Persistence from 23:00. Farm was OFF (0.076 MW).
            # NWP says sub-3 m/s. Very likely stays off.
            # But the evening showed brief gusts producing 5-7 MW even at
            # NWP ws=1.0. Those were transient and subsided by 23:00.
            # Persistence of the 23:00 state (off) is the safest bet.
            # Allow 0.3-0.5 MW buffer for metering noise.
            final[h] = 0.4
            logic = "persistence (farm OFF at 23h)"
        elif h == 3:
            # Transition hour. NWP says 3.2 m/s (at cut-in).
            # Conditional median at 3.0-3.5 = 1.80 MW.
            # Model says 2.68. Blend conservatively.
            final[h] = 0.5 * mp + 0.5 * cm
            logic = "50/50 model + cond_median"
        else:
            # Hours 4-8: NWP says above cut-in (3.75-4.75 m/s).
            # The conditional median is HIGHER than the model prediction
            # (the model under-predicts for this spring-overnight wind regime).
            # Use 60% model + 40% conditional median (as in v2).
            final[h] = 0.6 * mp + 0.4 * cm
            logic = "60% model + 40% cond_median"

        print(f"  {h:>2}h   {ws:>6.2f}  {mp:>6.2f}  {cm:>7.2f}  "
              f"{final[h]:>6.2f}   {logic}")

    final = np.clip(final, 0.0, CAPACITY_MW)

    print(f"\n  Summary:")
    print(f"    Mean: model={model_pred.mean():.3f}  v3={final.mean():.3f}  "
          f"diff={final.mean() - model_pred.mean():+.3f} MW")
    print(f"    Range: [{final.min():.2f}, {final.max():.2f}] MW")

    # Write
    out_df = out[["datetime", "hour"]].copy()
    out_df["forecast_mw"] = final
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT, index=False)
    print(f"\n  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
