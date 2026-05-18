"""Quick sanity check on the submission file."""
import numpy as np
import pandas as pd

preds = pd.read_csv(r"f:\Claude\win_d\submissions\archive\v0.1_lgbm_baseline.csv", header=None)[0].values
print(f"Rows: {len(preds)}")
print(f"Min: {preds.min():.4f}, Max: {preds.max():.4f}, Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
print(f"Any NaN: {np.any(np.isnan(preds))}")
print(f"Any < 0: {np.any(preds < 0)}")
print(f"Any > 90.09: {np.any(preds > 90.09)}")
print(f"Median: {np.median(preds):.4f}")
print(f"P10: {np.percentile(preds, 10):.4f}, P90: {np.percentile(preds, 90):.4f}")
