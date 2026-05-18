"""Post-process the 0-8h forecast with physics-based corrections.

Three targeted improvements for the 00:00-08:00 window:

  1. Sub-cut-in clamping: NWP wind < 3 m/s → clamp prediction to near-zero
  2. Physics-informed floor from manufacturer PC (don't predict above what
     the power curve says is physically possible at the given wind)
  3. Backtest-calibrated bias correction (the backtest showed +1.5 MW
     systematic over-prediction at night/calm)

Uses the existing 24-hour forecast as the base and applies corrections
only to hours 0-8.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TOTAL_TURBINES
from src.features.datasheet_power_curve import per_turbine_power_kw

INPUT  = _ROOT / "submissions" / "18_05_2026_forecast.csv"
OUTPUT = _ROOT / "submissions" / "18_05_2026_final_0_8.csv"

# NWP data for May 18 (from the test file inspection above)
# Sorted by hour ascending.
MAY18_NWP = {
    # hour: (wind_speed_120m, wind_speed_80m, wind_gusts_10m)
    0: (1.34, 1.68, 2.2),   # midnight — well below cut-in
    1: (2.52, 2.55, 2.3),   # below cut-in
    2: (2.02, 1.56, 2.4),   # below cut-in
    3: (3.20, 2.70, 2.5),   # at cut-in threshold
    4: (4.75, 4.61, 4.1),   # above cut-in, low power
    5: (4.60, 4.30, 5.0),   # same
    6: (4.40, 4.20, 4.8),   # same
    7: (3.75, 3.57, 4.9),   # barely above cut-in
    8: (4.53, 4.39, 5.1),   # above cut-in
}

# Farm parameters
N_ACTIVE = 24   # from the prediction output
AIR_DENSITY_MAY = 1.20  # typical May at this site (from training data: ~15-16°C, ~1004 hPa)

# May 17 evening context: wind was 0.76-1.90 m/s at 120m → farm was OFF
# May 17 23:00: 1.00 m/s. So at midnight the farm transitions from completely stopped.
MAY17_23_WS120 = 1.00


def manufacturer_pc_farm_mw(ws_hub: float, n_active: int = N_ACTIVE,
                            rho: float = AIR_DENSITY_MAY) -> float:
    """Farm-level MW from the Siemens Gamesa SG 3.4-132 datasheet."""
    ws_arr = np.array([ws_hub], dtype=float)
    rho_arr = np.array([rho], dtype=float)
    p_kw = per_turbine_power_kw(ws_arr, rho_arr)[0]
    return p_kw * n_active / 1000.0


def main():
    df = pd.read_csv(INPUT)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Filter to 0-8
    mask = df["hour"].between(0, 8)
    out = df[mask].copy().sort_values("hour").reset_index(drop=True)

    print("Before corrections (existing blend):")
    print(f"{'Hour':>5}  {'Forecast':>9}  {'NWP ws120':>9}  {'PC_MW':>7}  {'Action':>20}")
    print(f"  {'-'*60}")

    corrected = out["forecast_mw"].to_numpy().copy()

    for i, row in out.iterrows():
        h = int(row["hour"])
        ws120, ws80, gusts = MAY18_NWP[h]
        old_pred = corrected[i]

        # Physics: what does the datasheet say for this wind?
        pc_mw = manufacturer_pc_farm_mw(ws120)

        action = ""

        # Correction 1: Sub-cut-in clamping
        # At ws_120m < 3.0 m/s, turbines are physically stopped.
        # Even at ws = 2.5-3.0 m/s, production is near-zero (37 kW/turbine
        # at 3.0 m/s = 0.89 MW for 24 turbines).
        if ws120 < 3.0:
            # The farm was off at 23:00 (ws=1.0). Persistence says it
            # stayed off. The NWP says sub-3 m/s. Predict near-zero.
            corrected[i] = min(old_pred, 0.5)
            action = "SUB-CUTIN → 0.5"
        elif ws120 < 3.5:
            # Barely at cut-in. Datasheet says ~37-100 kW/turbine at 3-3.5 m/s.
            # With 24 turbines: 0.89-2.4 MW. Cap at PC + small margin.
            cap = pc_mw * 1.3  # 30% above PC to allow for gusts
            if old_pred > cap:
                corrected[i] = cap
                action = f"CUT-IN CAP → {cap:.2f}"
        else:
            # Above cut-in (ws > 3.5 m/s). Apply a physics ceiling:
            # don't predict more than PC × 1.5 (allows wake/gust uplift)
            cap = pc_mw * 1.5
            if old_pred > cap:
                corrected[i] = cap
                action = f"PC CAP → {cap:.2f}"

        # Correction 2: Backtest bias correction
        # The May 15-17 backtest showed +1.5 MW avg over-prediction during
        # calm hours. Apply a 0.8x shrinkage for hours 0-3 (calmest period).
        if h <= 3 and corrected[i] > 0.5:
            corrected[i] *= 0.8
            if action:
                action += " +shrink"
            else:
                action = "night shrink ×0.8"

        print(f"  {h:>2}h   {old_pred:>8.3f}  {ws120:>8.2f}   {pc_mw:>6.2f}  {action:>20}")

    out["forecast_mw"] = corrected.clip(0.0, CAPACITY_MW)

    print(f"\nAfter corrections:")
    print(f"{'Hour':>5}  {'Old':>8}  {'New':>8}  {'Diff':>8}")
    print(f"  {'-'*35}")
    old_vals = df.loc[mask, "forecast_mw"].sort_values(
        key=lambda s: df.loc[s.index, "hour"]
    ).to_numpy()
    for i, row in out.iterrows():
        h = int(row["hour"])
        print(f"  {h:>2}h   {old_vals[i]:>7.3f}  {corrected[i]:>7.3f}  "
              f"{corrected[i] - old_vals[i]:>+7.3f}")

    print(f"\n  Mean: old={old_vals.mean():.3f} MW  new={corrected.mean():.3f} MW  "
          f"diff={corrected.mean() - old_vals.mean():+.3f} MW")

    # Write final submission
    final = out[["datetime", "hour", "forecast_mw"]].copy()
    final["forecast_mw"] = corrected
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT, index=False)
    print(f"\n  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
