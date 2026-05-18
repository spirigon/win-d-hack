"""V108: PyTorch MLP neural network — genuinely different architecture from LGBM.

Key design:
  - Same top-90 feature set as V99 (corr dedup + GEM/ICON-G)
  - 4-layer MLP: 90 → 256 → 128 → 64 → 1 with BatchNorm + Dropout(0.3)
  - Sigmoid output → CF prediction in [0, 1]
  - MAE loss (direct alignment with nMAE metric)
  - Adam lr=5e-4, weight_decay=1e-4, cosine LR schedule
  - 5 seeds × 3 folds, CPU only (to avoid WDDM crash)
  - StandardScaler normalization per fold
  - Early stopping: val MAE patience=30 epochs, max 400 epochs

MLP provides smooth function approximation (vs LGBM's piecewise-constant
step functions) which should better fit the steep region of the power curve.

Outputs:
    data/processed/v108_oof.parquet
    submissions/archive/v108.0_mlp.csv
    submissions/archive/v108.1_mlp_lgbm_blend.csv   (50/50 with v99)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.loaders import load_valid_features
from src.data.outliers import identify_impossible_rows
from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.advanced_interactions import add_all_advanced_features, advanced_interaction_columns
from src.features.availability import add_walk_forward_availability
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.era5_v2 import merge_era5_v2
from src.features.extras import add_extra_features
from src.features.gem_icon_features import add_gem_icon_features, gem_icon_columns
from src.features.hub_height_features import add_hub_height_features, hub_height_columns
from src.features.multi_nwp_features import (
    build_nwp_consensus, merge_ecmwf_ifs, merge_gfs, multi_nwp_columns,
)
from src.features.nasa_features import merge_nasa_merra2, nasa_columns
from src.features.nwp_ensemble import merge_nwp_ensemble, nwp_ensemble_columns
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.seasonal_features import add_seasonal_features, seasonal_columns
from src.features.wake import add_wake_features, fit_wake_lookup
from src.inference.submission import write_submission
from src.models.lightgbm_model import LGBMConfig
from src.utils.seeding import set_global_seed

from src.training.train_v32_era5v2 import (
    LGBM_PARAMS, K, BLEND_WEIGHT_MW, FOLD_IDS, SEEDS_5,
    _load_train_raw, _add_pc, _merge_era5, _add_era5_rolling, _to_cf, _from_cf,
)

TRAIN_PATH  = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH  = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH   = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OOF_PATH    = _ROOT / "data" / "processed" / "v108_oof.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "archive" / "v108.0_mlp.csv"
OUTPUT_BLEND_PATH = _ROOT / "submissions" / "archive" / "v108.1_mlp_lgbm_blend.csv"
V99_OOF_PATH = _ROOT / "data" / "processed" / "v99_oof.parquet"
V99_SUB_PATH = _ROOT / "submissions" / "archive" / "v99.0_corr_dedup.csv"

MARCH_WEIGHT   = 3.0
SEEDS          = SEEDS_5
K              = 90
CORR_THRESHOLD = 0.999

# MLP hyperparameters
MLP_LR         = 5e-4
MLP_WEIGHT_DECAY = 1e-4
MLP_MAX_EPOCHS = 400
MLP_PATIENCE   = 30
MLP_BATCH_SIZE = 256
DEVICE         = "cpu"  # CPU to avoid WDDM GPU crash


class WindPowerMLP(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_mlp_single(
    X_tr: np.ndarray, y_tr: np.ndarray, sw_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, nn.Module, StandardScaler]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    # Fill NaN with column mean before scaling
    col_means = np.nanmean(X_tr, axis=0)
    nan_mask = np.isnan(X_tr)
    X_tr_filled = np.where(nan_mask, col_means[None, :], X_tr)
    nan_mask_va = np.isnan(X_va)
    X_va_filled = np.where(nan_mask_va, col_means[None, :], X_va)

    X_tr_s = scaler.fit_transform(X_tr_filled).astype(np.float32)
    X_va_s = scaler.transform(X_va_filled).astype(np.float32)
    y_tr_s = y_tr.astype(np.float32)
    y_va_s = y_va.astype(np.float32)
    sw_s = sw_tr.astype(np.float32)

    n = len(X_tr_s)
    model = WindPowerMLP(X_tr_s.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_MAX_EPOCHS)

    X_tr_t = torch.from_numpy(X_tr_s).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr_s).to(DEVICE)
    sw_t   = torch.from_numpy(sw_s).to(DEVICE)
    X_va_t = torch.from_numpy(X_va_s).to(DEVICE)
    y_va_t = torch.from_numpy(y_va_s).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(MLP_MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, MLP_BATCH_SIZE):
            idx = perm[start: start + MLP_BATCH_SIZE]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            wb = sw_t[idx]
            pred = model(xb)
            loss = (wb * torch.abs(pred - yb)).sum() / wb.sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = float(torch.mean(torch.abs(val_pred - y_va_t)).item())

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= MLP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        # Return validation predictions
        va_out = model(X_va_t).cpu().numpy()
    return va_out, model, scaler


def _predict_mlp(model: nn.Module, scaler: StandardScaler, X: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    model.eval()
    nan_mask = np.isnan(X)
    X_filled = np.where(nan_mask, col_means[None, :], X)
    X_s = scaler.transform(X_filled).astype(np.float32)
    X_t = torch.from_numpy(X_s).to(DEVICE)
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def _corr_dedup(feat_cols: list[str], X: np.ndarray, threshold: float) -> list[str]:
    n, p = X.shape
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X_filled = X.copy()
    for j in range(p):
        X_filled[nan_mask[:, j], j] = col_means[j]
    col_stds = X_filled.std(axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    X_normed = (X_filled - X_filled.mean(axis=0)) / col_stds
    corr = (X_normed.T @ X_normed) / n
    kept: list[int] = []
    for i in range(p):
        if not kept:
            kept.append(i)
            continue
        max_abs = float(np.max(np.abs(corr[i, kept])))
        if max_abs <= threshold:
            kept.append(i)
    return [feat_cols[i] for i in kept]


def _lgbm_fold_ensemble(X_tr, y_tr, X_va, y_va, X_test, feat_cols, ws_tr, sw, seeds):
    """Same LGBM ensemble as v99 (for the blend)."""
    regime_val, regime_test = {}, {}
    for name, (lo, hi) in [("low_0_7", (0, 7)), ("mid_4_12", (4, 12)), ("high_8_25", (8, 25))]:
        mask = (ws_tr >= lo) & (ws_tr < hi)
        weights = (np.where(mask, 2.0, 0.3) * sw).astype(np.float32)
        vp, tp = [], []
        for s in seeds:
            cfg = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": s})
            dt = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                             feature_name=feat_cols, free_raw_data=False)
            dv = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols, free_raw_data=False)
            b = lgb.train(
                cfg.to_params(), dt, num_boost_round=LGBM_PARAMS.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(LGBM_PARAMS.early_stopping_rounds, verbose=False)],
            )
            vp.append(b.predict(X_va, num_iteration=b.best_iteration))
            tp.append(b.predict(X_test, num_iteration=b.best_iteration))
        regime_val[name]  = np.mean(vp, axis=0)
        regime_test[name] = np.mean(tp, axis=0)
    return (
        np.mean(list(regime_val.values()), axis=0),
        np.mean(list(regime_test.values()), axis=0),
    )


def main() -> None:
    set_global_seed(42)

    print("=" * 72)
    print(f"V108: PyTorch MLP ({len(SEEDS)} seeds, K={K}, device={DEVICE})")
    print("=" * 72)

    print("\n[1/4] Loading + building features...")
    df_train = _load_train_raw(TRAIN_PATH)
    df_valid = load_valid_features(VALID_PATH)
    era5 = pd.read_parquet(ERA5_PATH)

    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    combined = build_features(combined, sort_by_time=False)
    combined = _merge_era5(combined, era5)
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)
    combined = _add_era5_rolling(combined)
    combined = merge_era5_v2(combined)
    combined = merge_nwp_ensemble(combined)
    combined = merge_nasa_merra2(combined)
    combined = merge_gfs(combined)
    combined = merge_ecmwf_ifs(combined)
    combined = build_nwp_consensus(combined)
    combined = add_all_advanced_features(combined)
    combined = add_hub_height_features(combined)
    combined = add_seasonal_features(combined)
    combined = add_gem_icon_features(combined)

    nwp_cols  = nwp_ensemble_columns(combined)
    nasa_cols = nasa_columns(combined)
    mnwp_cols = multi_nwp_columns(combined)
    adv_cols  = advanced_interaction_columns(combined)
    hub_cols  = hub_height_columns(combined)
    seas_cols = seasonal_columns(combined)
    gem_cols  = gem_icon_columns(combined)

    df_train_full_pre = combined[combined["_split"] == "train"].reset_index(drop=True)
    impossible = identify_impossible_rows(df_train_full_pre)
    is_impossible_full = pd.Series(False, index=combined.index)
    is_impossible_full.iloc[: len(df_train_full_pre)] = impossible.values
    combined["_is_impossible"] = is_impossible_full.values
    train_end = df_train_full_pre[TIMESTAMP_COL].max()

    sample_weight_full = (~is_impossible_full.to_numpy()[: len(df_train_full_pre)]).astype(np.float32)
    march_mask = df_train_full_pre[TIMESTAMP_COL].dt.month == 3
    sample_weight_full[march_mask] *= MARCH_WEIGHT

    print(f"[2/4] Feature selection probe (Fold-5, K={K})...")
    folds = default_folds()
    fold5 = folds[-1]

    probe_combined = add_walk_forward_availability(
        combined, train_end=fold5.train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=fold5.train_end,
    )
    probe_train = probe_combined[probe_combined["_split"] == "train"].reset_index(drop=True)

    train_idx, val_idx = split_indices(probe_train, fold5)
    fold_train_ = probe_train.iloc[train_idx]
    fit_data_ = fold_train_[~fold_train_["_is_impossible"]]
    pc_s_ = fit_sector_isotonic(fit_data_, n_sectors=8)
    pc_g_ = IsotonicPowerCurve().fit(fit_data_["v_eff"], fit_data_[TARGET_COL])
    w_ = fit_wake_lookup(fit_data_, n_sectors=16)
    df_t_ = _add_pc(fold_train_, pc_s_, pc_g_)
    df_t_ = add_wake_features(df_t_, w_)
    df_v_ = _add_pc(probe_train.iloc[val_idx], pc_s_, pc_g_)
    df_v_ = add_wake_features(df_v_, w_)

    all_extra_cols = list(dict.fromkeys(
        nwp_cols + nasa_cols + mnwp_cols + adv_cols + hub_cols + seas_cols + gem_cols
    ))
    feat_cols_all = [
        c for c in feature_columns(df_t_)
        if c not in ("_is_impossible", "_split", TARGET_COL) and c != "avail_underprod_mw"
    ]
    for col in all_extra_cols:
        if col in df_t_.columns and col not in feat_cols_all:
            feat_cols_all.append(col)

    seen_keys: dict[bytes, str] = {}
    dedup_cols: list[str] = []
    for col in feat_cols_all:
        try:
            key = df_t_[col].to_numpy(dtype=np.float32).tobytes()
            if key not in seen_keys:
                seen_keys[key] = col
                dedup_cols.append(col)
        except Exception:
            dedup_cols.append(col)
    feat_cols_all = dedup_cols
    X_probe = df_t_[feat_cols_all].to_numpy(dtype=np.float64)
    feat_cols_all = _corr_dedup(feat_cols_all, X_probe, CORR_THRESHOLD)
    print(f"  Feature pool after dedup: {len(feat_cols_all)}")

    a_tr_ = df_t_["active_turbines"].to_numpy(dtype=np.float32)
    y_t_cf_ = _to_cf(df_t_[TARGET_COL].to_numpy(dtype=np.float32), a_tr_)
    a_va_ = df_v_["active_turbines"].to_numpy(dtype=np.float32)
    y_v_cf_ = _to_cf(df_v_[TARGET_COL].to_numpy(dtype=np.float32), a_va_)
    sw_fold = sample_weight_full[train_idx]

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_ = lgb.Dataset(df_t_[feat_cols_all].to_numpy(dtype=np.float32),
                      label=y_t_cf_, weight=sw_fold,
                      feature_name=feat_cols_all, free_raw_data=False)
    dv_ = lgb.Dataset(df_v_[feat_cols_all].to_numpy(dtype=np.float32),
                      label=y_v_cf_, feature_name=feat_cols_all, free_raw_data=False)
    probe = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp.tolist(), strict=True), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]
    print(f"  Top-{K} selected  (top-3: {[n for n,_ in feat_imp[:3]]})")

    print(f"\n[3/4] Training MLP + LGBM per fold (seeds={SEEDS})...")
    test_mlp_per_fold: dict[int, np.ndarray] = {}
    test_lgbm_per_fold: dict[int, np.ndarray] = {}
    oof_records: list[dict] = []

    for fold_idx in FOLD_IDS:
        t0 = time.time()
        fold = folds[fold_idx]
        print(f"\n  === Fold {fold_idx + 1} ===")

        fc = add_walk_forward_availability(
            combined, train_end=fold.train_end,
            wind_col="wind_speed_120m", impossible_col="_is_impossible",
            window_days=30, freeze_after_ts=fold.train_end,
        )
        ft = fc[fc["_split"] == "train"].reset_index(drop=True)
        fv = fc[fc["_split"] == "valid"].reset_index(drop=True)

        train_idx, val_idx = split_indices(ft, fold)
        fold_train = ft.iloc[train_idx]
        fold_val   = ft.iloc[val_idx]
        fit_data   = fold_train[~fold_train["_is_impossible"]]
        pc_sector  = fit_sector_isotonic(fit_data, n_sectors=8)
        pc_global  = IsotonicPowerCurve().fit(fit_data["v_eff"], fit_data[TARGET_COL])
        wake       = fit_wake_lookup(fit_data, n_sectors=16)
        df_tr = _add_pc(fold_train, pc_sector, pc_global)
        df_tr = add_wake_features(df_tr, wake)
        df_va = _add_pc(fold_val, pc_sector, pc_global)
        df_va = add_wake_features(df_va, wake)
        df_te = _add_pc(fv, pc_sector, pc_global)
        df_te = add_wake_features(df_te, wake)
        for c in set(top_k) - set(df_te.columns):
            df_te[c] = 0.0

        active_tr = df_tr["active_turbines"].to_numpy(dtype=np.float32)
        active_va = df_va["active_turbines"].to_numpy(dtype=np.float32)
        y_tr_mw   = df_tr[TARGET_COL].to_numpy(dtype=np.float32)
        y_va_mw   = df_va[TARGET_COL].to_numpy(dtype=np.float32)
        y_tr_cf   = _to_cf(y_tr_mw, active_tr)
        y_va_cf   = _to_cf(y_va_mw, active_va)
        X_tr   = df_tr[top_k].to_numpy(dtype=np.float32)
        X_va   = df_va[top_k].to_numpy(dtype=np.float32)
        X_test = df_te[top_k].to_numpy(dtype=np.float32)
        ws_tr  = df_tr["wind_speed_120m"].to_numpy()
        sw     = sample_weight_full[train_idx]

        # MLP training (CF mode, one per seed)
        mlp_val_preds, mlp_test_preds = [], []
        col_means_tr = np.nanmean(X_tr, axis=0)
        for s in SEEDS:
            t_seed = time.time()
            val_pred_cf, model_s, scaler_s = _train_mlp_single(
                X_tr, y_tr_cf, sw, X_va, y_va_cf, seed=s,
            )
            test_pred_cf = _predict_mlp(model_s, scaler_s, X_test, col_means_tr)
            mlp_val_preds.append(val_pred_cf)
            mlp_test_preds.append(test_pred_cf)
            print(f"    Seed {s}: MLP val MAE={float(np.mean(np.abs(val_pred_cf - y_va_cf))):.4f}  "
                  f"({time.time()-t_seed:.0f}s)")

        mlp_val_cf = np.mean(mlp_val_preds, axis=0)
        mlp_test_cf = np.mean(mlp_test_preds, axis=0)
        mlp_val_mw = np.clip(_from_cf(mlp_val_cf, active_va), 0, CAPACITY_MW)
        mlp_nmae = float(normalized_mae(y_va_mw, mlp_val_mw))
        print(f"  Fold {fold_idx + 1}: MLP nMAE={mlp_nmae:.4f}%  ({time.time()-t0:.0f}s total)")
        test_mlp_per_fold[fold_idx] = mlp_test_cf

        # LGBM CF training (same as v99, for the blend)
        lgbm_val_cf, lgbm_test_cf = _lgbm_fold_ensemble(
            X_tr, y_tr_cf, X_va, y_va_cf, X_test, top_k, ws_tr, sw, SEEDS,
        )
        lgbm_val_mw = np.clip(_from_cf(lgbm_val_cf, active_va), 0, CAPACITY_MW)
        lgbm_nmae = float(normalized_mae(y_va_mw, lgbm_val_mw))
        print(f"  Fold {fold_idx + 1}: LGBM nMAE={lgbm_nmae:.4f}%")
        test_lgbm_per_fold[fold_idx] = lgbm_test_cf

        # 50/50 blend
        blend_val_cf = 0.5 * mlp_val_cf + 0.5 * lgbm_val_cf
        blend_val_mw = np.clip(_from_cf(blend_val_cf, active_va), 0, CAPACITY_MW)
        blend_nmae = float(normalized_mae(y_va_mw, blend_val_mw))
        print(f"  Fold {fold_idx + 1}: Blend(MLP+LGBM) nMAE={blend_nmae:.4f}%")

        ts_va = df_va[TIMESTAMP_COL].to_numpy()
        for i in range(len(y_va_mw)):
            oof_records.append({
                "fold": int(fold_idx + 1),
                "ts": pd.Timestamp(ts_va[i]),
                "target_mw": float(y_va_mw[i]),
                "active_turbines": float(active_va[i]),
                "ws_120": float(df_va["wind_speed_120m"].to_numpy()[i]),
                "mlp_cf": float(mlp_val_cf[i]),
                "lgbm_cf": float(lgbm_val_cf[i]),
                "blend_cf": float(blend_val_cf[i]),
            })

    oof_df = pd.DataFrame(oof_records)
    oof_df["pred_blend_mw"] = np.clip(
        _from_cf(oof_df["blend_cf"].to_numpy(), oof_df["active_turbines"].to_numpy()),
        0, CAPACITY_MW,
    )
    oof_df["pred_mlp_mw"] = np.clip(
        _from_cf(oof_df["mlp_cf"].to_numpy(), oof_df["active_turbines"].to_numpy()),
        0, CAPACITY_MW,
    )
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(OOF_PATH, index=False)
    print(f"\n  OOF saved: {OOF_PATH}")
    for fid in [3, 4, 5]:
        sub = oof_df[oof_df["fold"] == fid]
        nb = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_blend_mw"].to_numpy()))
        nm = float(normalized_mae(sub["target_mw"].to_numpy(), sub["pred_mlp_mw"].to_numpy()))
        print(f"    Fold {fid}: MLP={nm:.4f}%  Blend={nb:.4f}%")
    all_blend = float(normalized_mae(oof_df["target_mw"].to_numpy(), oof_df["pred_blend_mw"].to_numpy()))
    print(f"    All-fold blend nMAE: {all_blend:.4f}%")

    print("\n[4/4] Building final submissions...")
    test_combined = add_walk_forward_availability(
        combined, train_end=train_end,
        wind_col="wind_speed_120m", impossible_col="_is_impossible",
        window_days=30, freeze_after_ts=train_end,
    )
    df_valid_sorted = test_combined[test_combined["_split"] == "valid"].reset_index(drop=True)

    avg_mlp_cf  = np.mean([test_mlp_per_fold[i]  for i in FOLD_IDS], axis=0)
    avg_lgbm_cf = np.mean([test_lgbm_per_fold[i] for i in FOLD_IDS], axis=0)
    active_valid = df_valid_sorted["active_turbines"].to_numpy(dtype=np.float32)

    # MLP-only submission
    mlp_mw = np.clip(_from_cf(avg_mlp_cf, active_valid), 0.0, CAPACITY_MW).astype(np.float64)
    order = df_valid_sorted["_submission_row"].to_numpy().astype(int)
    ts_arr = df_valid_sorted[TIMESTAMP_COL].to_numpy()
    mlp_po, ts_po = np.empty_like(mlp_mw), np.empty_like(ts_arr)
    mlp_po[order] = mlp_mw
    ts_po[order]  = ts_arr
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(mlp_po, OUTPUT_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"  MLP submission: {OUTPUT_PATH}")

    # 50/50 MLP+LGBM blend
    blend_cf = 0.5 * avg_mlp_cf + 0.5 * avg_lgbm_cf
    blend_mw = np.clip(_from_cf(blend_cf, active_valid), 0.0, CAPACITY_MW).astype(np.float64)
    blend_po = np.empty_like(blend_mw)
    blend_po[order] = blend_mw
    write_submission(blend_po, OUTPUT_BLEND_PATH, expected_rows=len(df_valid), timestamps=ts_po)
    print(f"  Blend submission: {OUTPUT_BLEND_PATH}")
    print(f"  MLP mean: {mlp_mw.mean():.2f} MW  Blend mean: {blend_mw.mean():.2f} MW")
    print(f"  Submission saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
