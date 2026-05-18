"""Compare baseline vs tuned submissions for sanity."""
import numpy as np
import pandas as pd

base = pd.read_csv(r"f:\Claude\win_d\submissions\archive\v0.1_lgbm_baseline.csv", header=None)[0].values
tuned = pd.read_csv(r"f:\Claude\win_d\submissions\archive\v0.2_lgbm_tuned.csv", header=None)[0].values

print(f"Baseline: mean={base.mean():.3f}, median={np.median(base):.3f}, std={base.std():.3f}")
print(f"Tuned:    mean={tuned.mean():.3f}, median={np.median(tuned):.3f}, std={tuned.std():.3f}")
print(f"Pearson corr between baseline and tuned: {np.corrcoef(base, tuned)[0,1]:.4f}")
print(f"Mean abs diff: {np.mean(np.abs(base - tuned)):.3f} MW")
print(f"Max abs diff: {np.max(np.abs(base - tuned)):.3f} MW")
