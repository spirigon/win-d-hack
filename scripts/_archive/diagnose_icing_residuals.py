"""Check whether the current LGBM systematically over- or under-predicts
during icing-prone conditions on Fold-5 (Q1 2025).

If it over-predicts (pred > actual) during icing, a derate helps.
If it under-predicts, a derate hurts.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.loaders import load_train
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed
from src.training.train_v20_oof_blend import _add_pc, merge_era5, add_era5_rolling, to_cf, from_cf, CONFIG

TRAIN_PATH = ROOT / "data" / "raw" / "train_dataset.csv"
ERA5_PATH = ROOT / "data" / "external" / "era5_reanalysis.parquet"
TURBINE_RATED_MW = 3.465
K = 80


def main():
    set_global_seed(42)
    df_train = load_train(TRAIN_PATH)
    era5 = pd.read_parquet(ERA5_PATH)
    df_train["_split"] = "train"
    combined = df_train.copy()
    combined = build_features(combined, sort_by_time=True)
    combined = merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = add_era5_rolling(combined)
    df_train = combined.reset_index(drop=True)
    imp = identify_impossible_rows(df_train)
    df_train["_is_impossible"] = imp.values

    folds = default_folds()
    fold5 = folds[-1]
    tr_idx, va_idx = split_indices(df_train, fold5)
    fold_tr = df_train.iloc[tr_idx]
    fold_va = df_train.iloc[va_idx].copy()
    fit_data = fold_tr[~fold_tr["_is_impossible"]]

    pc_sector = fit_sector_isotonic(fit_data, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
    wake = fit_wake_lookup(fit_data, n_sectors=16)

    df_tr = _add_pc(fold_tr, pc_sector, pc_global)
    df_tr = add_wake_features(df_tr, wake)
    df_va = _add_pc(fold_va, pc_sector, pc_global)
    df_va = add_wake_features(df_va, wake)

    feat_cols = [c for c in feature_columns(df_tr) if c not in ("_is_impossible", "_split")]
    active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
    active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_cf = to_cf(df_tr[TARGET_COL].to_numpy(dtype=np.float32), active_tr)
    y_va_mw = df_va[TARGET_COL].to_numpy(dtype=np.float32)
    y_va_cf = to_cf(y_va_mw, active_va)
    X_tr_all = df_tr[feat_cols].to_numpy(dtype=np.float32)
    X_va_all = df_va[feat_cols].to_numpy(dtype=np.float32)

    # Probe for top-K.
    cfg0 = LGBMConfig(**{**CONFIG.__dict__, "seed": 42})
    dt = lgb.Dataset(X_tr_all, label=y_tr_cf, feature_name=feat_cols, free_raw_data=False)
    dv = lgb.Dataset(X_va_all, label=y_va_cf, feature_name=feat_cols, free_raw_data=False)
    probe = lgb.train(cfg0.to_params(), dt, num_boost_round=5000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
    fi = sorted(zip(feat_cols, probe.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    top_k = [n for n, _ in fi[:K]]
    X_tr = df_tr[top_k].to_numpy(dtype=np.float32)
    X_va = df_va[top_k].to_numpy(dtype=np.float32)
    ws_tr = df_tr["wind_speed_120m"].to_numpy()

    # Train avg3 specialists (3 seeds each).
    from src.training.train_v20_oof_blend import train_lgbm_spec_fold
    regime_preds = []
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = np.where(mask, 2.0, 0.3).astype(np.float32)
        val_p, _ = train_lgbm_spec_fold(
            X_tr, y_tr_cf, X_va, y_va_cf, X_va,  # use X_va as dummy test
            top_k, [42, 123, 456], weights
        )
        regime_preds.append(val_p)
    preds_cf = np.mean(regime_preds, axis=0)
    preds_mw = np.clip(from_cf(preds_cf, active_va), 0, CAPACITY_MW)

    # Compute residuals (actual - predicted).
    res = y_va_mw - preds_mw
    print(f"\nOverall: n={len(y_va_mw)} nMAE={np.abs(res).mean()/CAPACITY_MW*100:.4f}%")
    print(f"  Mean residual (y - pred): {res.mean():+.3f} MW")
    print(f"  Mean |residual|:          {np.abs(res).mean():.3f} MW")

    # Subset by icing regime.
    t80 = df_va["temperature_80m"].to_numpy()
    precip = df_va["rain"].to_numpy() + df_va["showers"].to_numpy() + df_va["snowfall"].to_numpy()
    snowfall = df_va["snowfall"].to_numpy()
    cloud = df_va["cloud_cover_low"].to_numpy()

    def analyze(mask, label):
        n = mask.sum()
        if n == 0:
            print(f"\n{label}: n=0")
            return
        print(f"\n{label}: n={n} ({n/len(mask)*100:.1f}%)")
        r = res[mask]
        a = y_va_mw[mask]
        p = preds_mw[mask]
        print(f"  Mean actual:   {a.mean():.3f} MW")
        print(f"  Mean pred:     {p.mean():.3f} MW")
        print(f"  Mean residual: {r.mean():+.3f} MW  (positive = model under-predicts)")
        print(f"  Mean |residual|:  {np.abs(r).mean():.3f} MW")
        # nMAE over this subset (same denominator as contest).
        print(f"  Subset nMAE:   {np.abs(r).mean() / CAPACITY_MW * 100:.4f}%")
        # Hypothetical effect of a multiplicative derate.
        for factor in [0.85, 0.90, 0.95]:
            new_pred = p * factor
            new_res = a - new_pred
            new_nmae = np.abs(new_res).mean() / CAPACITY_MW * 100
            delta = new_nmae - np.abs(r).mean() / CAPACITY_MW * 100
            print(f"    If derate by {factor}: subset nMAE = {new_nmae:.4f}% (Δ={delta:+.4f})")

    analyze((t80 < 0) & (precip > 0), "T<0 & precip>0 (classic icing)")
    analyze((t80 < 2) & (precip > 0), "T<2 & precip>0 (rime ice)")
    analyze((t80 < 0) & (snowfall > 0), "T<0 & snow>0")
    analyze((t80 < 0), "T<0 (any)")
    analyze((t80 < 2) & (precip > 0) & (cloud > 80), "T<2 & precip & high cloud (heavy rime)")

    # Also check: when model over-predicts (residual < 0), what's happening?
    over_pred_mask = res < -5  # model predicted >5 MW more than actual
    print(f"\nLarge over-predictions (pred-actual > 5): n={over_pred_mask.sum()}")
    if over_pred_mask.sum() > 0:
        print(f"  Mean T80: {t80[over_pred_mask].mean():.2f}")
        print(f"  Pct with T<0: {(t80[over_pred_mask] < 0).mean()*100:.1f}%")
        print(f"  Pct with precip: {(precip[over_pred_mask] > 0).mean()*100:.1f}%")
        print(f"  Pct with T<2 & precip: {((t80[over_pred_mask] < 2) & (precip[over_pred_mask] > 0)).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
