"""Check distribution of valid-set v_eff near cut-in threshold."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.features.physics import compute_air_density, compute_v_eff

valid = pd.read_csv(_ROOT / "data" / "raw" / "valid_features.csv")
density = compute_air_density(valid["pressure_msl"], valid["temperature_80m"])
v_eff = compute_v_eff(valid["wind_speed_120m"], density)
print(f"Valid rows: {len(valid)}")
print(f"v_eff stats: min={v_eff.min():.2f}, max={v_eff.max():.2f}, mean={v_eff.mean():.2f}")

bins = [(0, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 5), (5, 7), (7, 25)]
print("\nDistribution of v_eff in valid set:")
for lo, hi in bins:
    n = ((v_eff >= lo) & (v_eff < hi)).sum()
    print(f"  [{lo:.1f}, {hi:.1f}): {n:4d} rows ({n/len(valid)*100:.1f}%)")

# Near cut-in (2-4 m/s).
near_cut = ((v_eff >= 2) & (v_eff <= 4)).sum()
print(f"\nRows near cut-in [2, 4]: {near_cut} ({near_cut/len(valid)*100:.2f}%)")
print(f"  Potential impact on nMAE: at most ~{near_cut/len(valid)*100*5/90:.3f}% absolute (if we mispredict by 5 MW in these rows)")

# In training, how does power transition around cut-in?
train = pd.read_csv(_ROOT / "data" / "raw" / "train_dataset.csv")
train["density"] = compute_air_density(train["pressure_msl"], train["temperature_80m"])
train["v_eff"] = compute_v_eff(train["wind_speed_120m"], train["density"])
target = "\u0412\u044b\u0440\u0430\u0431\u043e\u0442\u043a\u0430. \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0440\u0430\u0441\u0447\u0435\u0442"
print("\nTraining data: mean power by v_eff bin (near cut-in):")
for lo, hi in [(1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 5)]:
    mask = (train["v_eff"] >= lo) & (train["v_eff"] < hi)
    n = mask.sum()
    if n > 0:
        mean_p = train.loc[mask, target].mean()
        median_p = train.loc[mask, target].median()
        p25 = train.loc[mask, target].quantile(0.25)
        p75 = train.loc[mask, target].quantile(0.75)
        print(f"  [{lo:.1f}, {hi:.1f}): n={n:4d}, mean={mean_p:.2f}, median={median_p:.2f}, p25={p25:.2f}, p75={p75:.2f}")
