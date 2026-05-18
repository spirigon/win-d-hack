"""Diagnose why nMAE is ~8% on the Apr-May 2026 validation set.

Investigates NWP wind speed bias, physical model accuracy, error patterns by
wind speed bin and hour of day.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from src.features.pipeline import build_features
from src.features.datasheet_power_curve import (
    add_datasheet_power_features,
    farm_theoretical_power_mw,
    per_turbine_power_kw,
    WS_GRID,
)
from src.features.physics import fit_sector_isotonic, IsotonicPowerCurve, SectorIsotonicPowerCurve
from src.features.wake import add_wake_features, fit_wake_lookup
from src.data.schema import (
    CAPACITY_MW,
    TARGET_COL,
    TIMESTAMP_COL,
    TURBINES_IN_MAINTENANCE_COL,
    TOTAL_TURBINES,
)
from src.eval.metrics import normalized_mae

TURBINE_RATED_MW = 3.465

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df_train_raw = pd.read_csv("data/raw/train_dataset.csv", parse_dates=[TIMESTAMP_COL])
print(f"Train shape: {df_train_raw.shape}")
print(f"Train date range: {df_train_raw[TIMESTAMP_COL].min()} -> {df_train_raw[TIMESTAMP_COL].max()}")

df_may_raw = pd.read_csv("data/raw/18.05_test_dataset.csv", parse_dates=[TIMESTAMP_COL])
print(f"\nMay file shape: {df_may_raw.shape}")
print(f"May date range: {df_may_raw[TIMESTAMP_COL].min()} -> {df_may_raw[TIMESTAMP_COL].max()}")

# Split May file into historical (power known) and test (power NaN).
may_hist_mask = df_may_raw[TARGET_COL].notna()
df_may_hist = df_may_raw[may_hist_mask].copy()
df_may_test = df_may_raw[~may_hist_mask].copy()
print(f"\nMay historical rows (Apr 1 - May 17, power known): {len(df_may_hist)}")
print(f"May test rows (May 18, power NaN): {len(df_may_test)}")
print(f"Val power stats: mean={df_may_hist[TARGET_COL].mean():.2f} MW, "
      f"std={df_may_hist[TARGET_COL].std():.2f} MW, "
      f"min={df_may_hist[TARGET_COL].min():.2f}, max={df_may_hist[TARGET_COL].max():.2f}")

# ---------------------------------------------------------------------------
# 2. Build features on combined train + val to get rolling features correctly
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BUILDING FEATURES (combined train + val)")
print("=" * 70)

df_combined_raw = pd.concat([df_train_raw, df_may_hist], ignore_index=True)
print(f"Combined shape before features: {df_combined_raw.shape}")

df_combined = build_features(df_combined_raw, sort_by_time=True)
print(f"Combined shape after build_features: {df_combined.shape}")

# Add datasheet power features.
df_combined = add_datasheet_power_features(df_combined)
print(f"Datasheet columns added: {[c for c in df_combined.columns if c.startswith('ds_')]}")

# Split back into train and val.
val_timestamps = set(df_may_hist[TIMESTAMP_COL])
is_val = df_combined[TIMESTAMP_COL].isin(val_timestamps)
df_train = df_combined[~is_val].copy()
df_val = df_combined[is_val].copy()

print(f"\nTrain rows after split: {len(df_train)}")
print(f"Val rows after split: {len(df_val)}")
print(f"Val date range: {df_val[TIMESTAMP_COL].min()} -> {df_val[TIMESTAMP_COL].max()}")

# Verify target is present in val.
print(f"Val target notna count: {df_val[TARGET_COL].notna().sum()}")

# ---------------------------------------------------------------------------
# 3. Fit models on train only
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FITTING MODELS ON TRAIN DATA")
print("=" * 70)

# Wake lookup.
wake_lookup = fit_wake_lookup(df_train)
print("Wake lookup fitted.")

# Add wake features to full combined set.
df_train = add_wake_features(df_train, wake_lookup)
df_val = add_wake_features(df_val, wake_lookup)

# Check key column.
if "ds_consensus_wake_corrected" not in df_val.columns:
    print("\nERROR: 'ds_consensus_wake_corrected' not found!")
    print("Available columns:")
    for c in sorted(df_val.columns):
        print(f"  {c}")
    sys.exit(1)

print(f"'ds_consensus_wake_corrected' column present: OK")

# Sector isotonic on train only.
sector_isotonic = fit_sector_isotonic(df_train)
print("Sector isotonic power curve fitted.")

# ---------------------------------------------------------------------------
# 4. Predictions on val set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING PREDICTIONS ON VAL SET")
print("=" * 70)

y_true = df_val[TARGET_COL].to_numpy()
active_turbines = df_val["active_turbines"].to_numpy()

# Physical model: wake-corrected datasheet.
y_phys = df_val["ds_consensus_wake_corrected"].clip(0, CAPACITY_MW).to_numpy()

# Sector isotonic.
v_eff_val = df_val["v_eff"].to_numpy()
dir_deg_val = df_val["wind_direction_120m"].to_numpy() * 1000.0
y_isotonic = sector_isotonic.predict(v_eff_val, dir_deg_val)

# Naive baseline: just the un-wake-corrected consensus.
y_consensus_raw = df_val["ds_consensus_farm_mw"].clip(0, CAPACITY_MW).to_numpy()

# ---------------------------------------------------------------------------
# 4a. Overall nMAE
# ---------------------------------------------------------------------------
print("\n--- Overall nMAE ---")
nmae_phys = normalized_mae(y_true, y_phys)
nmae_iso = normalized_mae(y_true, y_isotonic)
nmae_raw = normalized_mae(y_true, y_consensus_raw)

print(f"Physical (wake-corrected datasheet) nMAE : {nmae_phys:.3f}%")
print(f"Sector isotonic nMAE                     : {nmae_iso:.3f}%")
print(f"Raw consensus (no wake correction) nMAE  : {nmae_raw:.3f}%")
print(f"\nVal mean power : {y_true.mean():.3f} MW")
print(f"Val std power  : {y_true.std():.3f} MW")
print(f"Phys mean pred : {y_phys.mean():.3f} MW")
print(f"Isotonic mean  : {y_isotonic.mean():.3f} MW")

# Mean signed error (positive = model overpredicts).
print(f"\nPhysical mean signed error (pred - true): {(y_phys - y_true).mean():.3f} MW")
print(f"Isotonic mean signed error (pred - true): {(y_isotonic - y_true).mean():.3f} MW")

# ---------------------------------------------------------------------------
# 4b. Error by wind speed bin (ws_120m)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ERROR BY WIND SPEED BIN (ws_120m)")
print("=" * 70)

ws_val = df_val["wind_speed_120m"].to_numpy()
bin_edges = [0, 3, 5, 7, 10, 13, 100]
bin_labels = ["0-3", "3-5", "5-7", "7-10", "10-13", "13+"]

print(f"\n{'Bin (m/s)':<12} {'Count':>6} {'MeanPwr':>8} {'PhysBias':>10} {'PhysMAE':>9} {'IsoBias':>10} {'IsoMAE':>8}")
print("-" * 68)
for lo, hi, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
    mask = (ws_val >= lo) & (ws_val < hi)
    if mask.sum() == 0:
        continue
    yt = y_true[mask]
    yp = y_phys[mask]
    yi = y_isotonic[mask]
    bias_phys = (yp - yt).mean()
    mae_phys = np.abs(yp - yt).mean()
    bias_iso = (yi - yt).mean()
    mae_iso = np.abs(yi - yt).mean()
    print(
        f"{label:<12} {mask.sum():>6} {yt.mean():>8.2f} "
        f"{bias_phys:>+10.2f} {mae_phys:>9.2f} "
        f"{bias_iso:>+10.2f} {mae_iso:>8.2f}"
    )

# ---------------------------------------------------------------------------
# 4c. Error by hour of day
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MEAN SIGNED ERROR BY HOUR OF DAY (Physical Model)")
print("=" * 70)

hour_val = df_val[TIMESTAMP_COL].dt.hour.to_numpy()
phys_err = y_phys - y_true
iso_err = y_isotonic - y_true

print(f"\n{'Hour':>5} {'Count':>6} {'PhysBias':>10} {'IsoBias':>10} {'MeanPwr':>8}")
print("-" * 45)
for h in range(24):
    mask = hour_val == h
    if mask.sum() == 0:
        continue
    print(
        f"{h:>5} {mask.sum():>6} {phys_err[mask].mean():>+10.3f} "
        f"{iso_err[mask].mean():>+10.3f} {y_true[mask].mean():>8.3f}"
    )

# ---------------------------------------------------------------------------
# 4d. Error by month
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ERROR BY MONTH")
print("=" * 70)

month_val = df_val[TIMESTAMP_COL].dt.month.to_numpy()
print(f"\n{'Month':>6} {'Count':>6} {'MeanPwr':>8} {'PhysNMAE':>10} {'IsoNMAE':>9}")
print("-" * 45)
for m in sorted(np.unique(month_val)):
    mask = month_val == m
    if mask.sum() < 5:
        continue
    nm_p = normalized_mae(y_true[mask], y_phys[mask])
    nm_i = normalized_mae(y_true[mask], y_isotonic[mask])
    print(f"{m:>6} {mask.sum():>6} {y_true[mask].mean():>8.2f} {nm_p:>10.3f} {nm_i:>9.3f}")

# ---------------------------------------------------------------------------
# 4e. NWP wind speed bias: implied wind speed from actual power
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("NWP WIND SPEED BIAS ANALYSIS")
print("=" * 70)

ws_120m_val = df_val["wind_speed_120m"].to_numpy()
density_val = df_val["air_density"].to_numpy()
active_val = df_val["active_turbines"].to_numpy()
power_mw_val = y_true

# For each row: compute per-turbine power, then invert power curve to get
# implied wind speed.
print("\nComputing implied wind speed via binary search on power curve...")

def inverse_power_curve(
    per_turbine_kw_target: np.ndarray,
    air_density: np.ndarray,
    ws_lo: float = 3.0,
    ws_hi: float = 25.0,
    n_iter: int = 30,
) -> np.ndarray:
    """Binary search: find ws that produces per_turbine_kw_target at given density."""
    lo = np.full(len(per_turbine_kw_target), ws_lo)
    hi = np.full(len(per_turbine_kw_target), ws_hi)
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        pw_mid = per_turbine_power_kw(mid, air_density)
        lo = np.where(pw_mid < per_turbine_kw_target, mid, lo)
        hi = np.where(pw_mid >= per_turbine_kw_target, mid, hi)
    return (lo + hi) / 2.0


# Filter: power > 0, not zero-output, not curtailed (actual < physical * 1.3).
# Also avoid the flat top (rated power region where inversion is ill-conditioned).
RATED_KW = TURBINE_RATED_MW * 1000.0

per_turb_kw_actual = np.where(
    active_val > 0,
    (power_mw_val * 1000.0) / np.maximum(active_val, 1),
    np.nan,
)
per_turb_phys_kw = per_turbine_power_kw(ws_120m_val, density_val)

bias_mask = (
    (power_mw_val > 2.0)           # non-trivial power
    & (per_turb_kw_actual < RATED_KW * 0.95)  # not in flat rated region
    & (power_mw_val < y_phys * 1.3)           # not severely curtailed above physical
    & (per_turb_kw_actual < RATED_KW)         # below rated
    & np.isfinite(per_turb_kw_actual)
)
print(f"Rows used for bias analysis: {bias_mask.sum()} of {len(bias_mask)}")

if bias_mask.sum() > 0:
    impl_ws = inverse_power_curve(
        per_turb_kw_actual[bias_mask],
        density_val[bias_mask],
    )
    nwp_ws = ws_120m_val[bias_mask]
    # Positive bias = NWP overestimates wind speed vs what physics says.
    bias_ws = nwp_ws - impl_ws

    print(f"\nNWP ws_120m mean          : {nwp_ws.mean():.3f} m/s")
    print(f"Implied ws (from actual power): {impl_ws.mean():.3f} m/s")
    print(f"Mean NWP bias (NWP - implied) : {bias_ws.mean():.3f} m/s  "
          f"({'NWP overestimates' if bias_ws.mean() > 0 else 'NWP underestimates'} wind)")
    print(f"Std of bias               : {bias_ws.std():.3f} m/s")
    print(f"Median bias               : {np.median(bias_ws):.3f} m/s")

    # Bias by wind speed bin (using NWP ws).
    print(f"\n--- NWP Wind Speed Bias by Bin ---")
    print(f"{'Bin (m/s)':<12} {'Count':>6} {'NWP_ws':>8} {'Implied_ws':>11} {'Bias':>8}")
    print("-" * 50)
    for lo_b, hi_b, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
        m2 = (nwp_ws >= lo_b) & (nwp_ws < hi_b)
        if m2.sum() < 3:
            continue
        print(
            f"{label:<12} {m2.sum():>6} {nwp_ws[m2].mean():>8.2f} "
            f"{impl_ws[m2].mean():>11.2f} {bias_ws[m2].mean():>+8.3f}"
        )

    # Bias by hour.
    print(f"\n--- NWP Wind Speed Bias by Hour ---")
    hour_bias = hour_val[bias_mask]
    print(f"{'Hour':>5} {'Count':>6} {'Bias':>8}")
    for h in range(24):
        mh = hour_bias == h
        if mh.sum() < 3:
            continue
        print(f"{h:>5} {mh.sum():>6} {bias_ws[mh].mean():>+8.3f}")

# ---------------------------------------------------------------------------
# 4f. Turbine maintenance distribution on val
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TURBINE AVAILABILITY ON VAL SET")
print("=" * 70)

maint_col = TURBINES_IN_MAINTENANCE_COL
if maint_col in df_val.columns:
    maint_val = df_val[maint_col].to_numpy()
    print(f"Turbines in maintenance: mean={maint_val.mean():.2f}, "
          f"max={maint_val.max()}, min={maint_val.min()}")
    unique, counts = np.unique(maint_val, return_counts=True)
    print("Distribution:")
    for u, c in zip(unique, counts):
        print(f"  {int(u)} turbines in maintenance: {c} hours ({c/len(maint_val)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 4g. Data quality check: NaN rates in val
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VAL SET COLUMN NaN RATES (key columns)")
print("=" * 70)

key_cols = [
    "wind_speed_120m", "wind_speed_80m", "wind_speed_180m",
    "wind_direction_120m", "air_density", "v_eff",
    "ds_80m_farm_mw", "ds_120m_farm_mw", "ds_consensus_farm_mw",
    "ds_consensus_wake_corrected", TARGET_COL,
]
for col in key_cols:
    if col in df_val.columns:
        n_nan = df_val[col].isna().sum()
        pct = n_nan / len(df_val) * 100
        print(f"  {col:<40s}: {n_nan:>4} NaN ({pct:.1f}%)")

# ---------------------------------------------------------------------------
# 4h. Sample head of val set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VAL SET HEAD (key columns)")
print("=" * 70)

display_cols = [
    TIMESTAMP_COL, "wind_speed_120m", "wind_direction_120m",
    "air_density", "active_turbines", "ds_consensus_farm_mw",
    "ds_consensus_wake_corrected", TARGET_COL,
]
display_cols = [c for c in display_cols if c in df_val.columns]
print(df_val[display_cols].head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Val set: {len(df_val)} rows, "
      f"{df_val[TIMESTAMP_COL].min().date()} to {df_val[TIMESTAMP_COL].max().date()}")
print(f"Physical model (wake-corrected datasheet) nMAE : {nmae_phys:.3f}%")
print(f"Sector isotonic nMAE                           : {nmae_iso:.3f}%")
print(f"Raw consensus (no wake) nMAE                   : {nmae_raw:.3f}%")
if bias_mask.sum() > 0:
    print(f"NWP ws_120m bias vs implied                    : {bias_ws.mean():+.3f} m/s "
          f"({'overestimate' if bias_ws.mean() > 0 else 'underestimate'})")
print("=" * 70)
