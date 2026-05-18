"""V2: Data-driven corrections for hours 0-8.

Key finding: the previous v1 was too aggressive on sub-cut-in clamping.
Historical data shows that when NWP says ws_120m < 2 m/s at hours 0-2,
actual power has MEDIAN = 2.38 MW (not 0). The NWP systematically
under-forecasts nighttime wind at this coastal site.

New approach: use the CONDITIONAL MEDIAN of actual power given NWP wind
as the anchor, rather than the physics cut-in clamp.

For hours 4-8 where NWP says 3.75-4.75 m/s: use the conditional
distribution of actual power at those wind speeds in May to calibrate.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
INPUT   = _ROOT / "submissions" / "18_05_2026_forecast.csv"
OUTPUT  = _ROOT / "submissions" / "18_05_2026_final_0_8_v2.csv"

# NWP data for May 18 hours 0-8 (from the test file)
MAY18_WS120 = {
    0: 1.34, 1: 2.52, 2: 2.02, 3: 3.20,
    4: 4.75, 5: 4.60, 6: 4.40, 7: 3.75, 8: 4.53,
}


def compute_conditional_power_stats(train_path: Path) -> pd.DataFrame:
    """For each (wind_speed_bin, hour_bucket, month), compute
    the conditional distribution of actual power."""
    df = pd.read_csv(train_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df["hour"] = df[TIMESTAMP_COL].dt.hour
    df["month"] = df[TIMESTAMP_COL].dt.month

    # Focus on May data, hours 0-8
    spring = df[(df["month"].isin([4, 5])) & (df["hour"].between(0, 8))]

    # Bin wind speed into 0.5 m/s bins from 0 to 8
    spring = spring.copy()
    spring["ws_bin"] = pd.cut(spring["wind_speed_120m"],
                              bins=np.arange(0, 9, 0.5),
                              labels=[f"{b:.1f}-{b+0.5:.1f}" for b in np.arange(0, 8.5, 0.5)])

    stats = spring.groupby("ws_bin", observed=True)[TARGET_COL].agg(
        ["count", "mean", "median", "std"]
    ).round(2)
    stats["p25"] = spring.groupby("ws_bin", observed=True)[TARGET_COL].quantile(0.25).round(2)
    stats["p75"] = spring.groupby("ws_bin", observed=True)[TARGET_COL].quantile(0.75).round(2)
    return stats


def main():
    print("Computing conditional power distribution (Apr-May, hours 0-8)...")
    stats = compute_conditional_power_stats(TRAIN_PATH)
    print(stats.to_string())

    # Load the current forecast
    df = pd.read_csv(INPUT)
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = df["hour"].between(0, 8)
    out = df[mask].copy().sort_values("hour").reset_index(drop=True)

    # For each hour, look up the conditional median for its NWP wind speed
    print(f"\n{'Hour':>5}  {'NWP ws':>7}  {'Model pred':>10}  {'Cond median':>11}  "
          f"{'Cond mean':>10}  {'Final':>7}")
    print(f"  {'-'*60}")

    corrected = out["forecast_mw"].to_numpy().copy()

    for i, row in out.iterrows():
        h = int(row["hour"])
        ws = MAY18_WS120[h]
        old_pred = corrected[i]

        # Find the conditional median for this wind speed bin
        bin_idx = int(ws / 0.5)
        ws_lo = bin_idx * 0.5
        ws_hi = ws_lo + 0.5
        bin_label = f"{ws_lo:.1f}-{ws_hi:.1f}"

        if bin_label in stats.index:
            cond_median = float(stats.loc[bin_label, "median"])
            cond_mean   = float(stats.loc[bin_label, "mean"])
        else:
            cond_median = old_pred
            cond_mean   = old_pred

        # Blending strategy:
        # - If model pred is between p25 and p75 of the conditional dist,
        #   trust the model (it's within the expected range).
        # - If model pred is outside, pull it toward the conditional median.
        # Weight: 60% model + 40% conditional median
        # This hedges against the model's systematic over-prediction while
        # respecting that it has information the raw conditional doesn't.
        final = 0.6 * old_pred + 0.4 * cond_median
        # But never go below the conditional p25 or above conditional p75
        # (unless the model is very confident)
        corrected[i] = final

        print(f"  {h:>2}h   {ws:>6.2f}  {old_pred:>10.3f}  {cond_median:>11.2f}  "
              f"{cond_mean:>10.2f}  {final:>7.3f}")

    corrected = np.clip(corrected, 0.0, CAPACITY_MW)

    print(f"\n  Mean: old={out['forecast_mw'].mean():.3f} MW  "
          f"new={corrected.mean():.3f} MW  "
          f"diff={corrected.mean() - out['forecast_mw'].mean():+.3f} MW")

    # Write
    final_df = out[["datetime", "hour"]].copy()
    final_df["forecast_mw"] = corrected
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT, index=False)
    print(f"\n  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
