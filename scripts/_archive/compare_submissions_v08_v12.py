"""Compare v0.8 vs v1.2 submissions: how similar are they?"""
import numpy as np
import pandas as pd

v08 = pd.read_csv(r"f:\Claude\win_d\submissions\archive\v0.8.csv", header=None)[0].values
v12 = pd.read_csv(r"f:\Claude\win_d\submissions\archive\v1.2_final.csv", header=None)[0].values

print(f"v0.8 rows: {len(v08)}, stats: mean={v08.mean():.3f}, std={v08.std():.3f}")
print(f"v1.2 rows: {len(v12)}, stats: mean={v12.mean():.3f}, std={v12.std():.3f}")
print(f"Correlation: {np.corrcoef(v08, v12)[0,1]:.6f}")
print(f"Mean abs diff: {np.mean(np.abs(v08 - v12)):.3f} MW")
print(f"Max abs diff: {np.max(np.abs(v08 - v12)):.3f} MW")
print(f"Rows with diff > 2 MW: {(np.abs(v08 - v12) > 2).sum()}")
print(f"Rows with diff > 5 MW: {(np.abs(v08 - v12) > 5).sum()}")
