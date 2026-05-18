"""Predict wind power for May 18, 2026 -- 24 hourly values.

Training data  : train_dataset.csv (2022-2025) + 18.05 historical rows (Apr 1 - May 17 2026)
ERA5 data      : extended to 2026-05-18 (see fetch_era5.py)
Test target    : 24 rows with NaN power in 18.05_test_dataset.csv (hours 0-23, May 18)

Ensemble
--------
1. LightGBM CF target  -- 3 regime specialists x 3 seeds (proven pipeline from train_best.py)
2. LightGBM MW target  -- same structure
3. ResNet MLP CF target -- residual blocks + physics-aware scaling (new DL model)

Final blend: 40% LGBM_CF + 40% LGBM_MW + 20% ResNet_MLP

Literature basis
----------------
- energies-18-00350 : LSTM best (MAPE 8.10%) vs linear 12.81%; CNN-LSTM hybrid outperforms RF/SVR
- wind-05-00029-v2  : CNN-LSTM with attention dominates 2020-2024 DNN reviews for short-term WPF;
                      hybrid architectures consistently beat single-model approaches
- Key insight: for 24-hour direct regression (no autoregression), residual MLP with
  physics-informed output is the most stable DL choice; blended with LGBM for robustness

Run
---
    cd F:\\Claude\\win_d
    python scripts/predict_18_05.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.outliers import identify_impossible_rows
from src.data.schema import (
    CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TURBINES_IN_MAINTENANCE_COL,
    TOTAL_TURBINES,
)
from src.data.splits import default_folds, split_indices
from src.eval.metrics import normalized_mae
from src.features.datasheet_power_curve import add_datasheet_power_features
from src.features.extras import add_extra_features
from src.features.pipeline import build_features, feature_columns
from src.features.physics import IsotonicPowerCurve, fit_sector_isotonic
from src.features.wake import add_wake_features, fit_wake_lookup
from src.models.lightgbm_model import LGBMConfig
from src.models.tft_model import (
    PhysicsInformedLoss,
    make_day_sequences,
    train_tft,
    predict_tft_day,
)
from src.utils.seeding import set_global_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH  = _ROOT / "data" / "raw" / "train_dataset.csv"
MAY_PATH    = _ROOT / "data" / "raw" / "18.05_test_dataset.csv"
ERA5_PATH   = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
OUTPUT_PATH = _ROOT / "submissions" / "18_05_2026_forecast.csv"

# ---------------------------------------------------------------------------
# Hyperparameters (Optuna-tuned from train_best.py v29)
# ---------------------------------------------------------------------------
LGBM_PARAMS = LGBMConfig(
    num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844,
    feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3,
    lambda_l1=0.253, lambda_l2=0.00971,
    num_boost_round=5000, early_stopping_rounds=200, log_period=0,
)
SEEDS        = [42, 123, 456]
K            = 80
TURBINE_RATED_MW = 3.465
BLEND_CF     = 0.30   # LGBM CF weight
BLEND_MW     = 0.30   # LGBM MW weight
BLEND_MLP    = 0.10   # ResNet MLP weight (physics-informed loss)
BLEND_TFT    = 0.30   # WindTFT P50 weight

# Training / validation splits
# Probe val: full 2025 (more reliable than Apr-May 47d holdout; avoids distribution
# shift — the Apr-May 2026 data are then included in LGBM training).
PROBE_VAL_START = pd.Timestamp("2025-01-01")
# Seasonal filter: train only on March-May from all years to reduce seasonal distribution
# shift (May wind patterns differ from winter/autumn; ~10k rows, avoids overfitting).
SPRING_MONTHS   = {3, 4, 5}
# ES val: March-May 2025 (same season as test day, most recent full spring before 2026)
SPRING_ES_START = pd.Timestamp("2025-03-01")
SPRING_ES_END   = pd.Timestamp("2025-06-01")

# Recency weights for LGBM (2026 data matches test distribution; weight it more)
_YEAR_WEIGHTS = {2022: 1.0, 2023: 1.5, 2024: 2.0, 2025: 3.0}
_YEAR_WEIGHT_2026 = 6.0

# ERA5 is excluded: for May 18 the Open-Meteo ERA5 archive returns ECMWF forecast
# data (not true reanalysis - ~5d lag), which correlates poorly with actual power
# (r=0.548 vs NWP r=0.819 on backtest). Training and prediction use NWP-only features.
USE_ERA5     = False

# MLP hyperparams
MLP_HIDDEN   = 256
MLP_DROPOUT  = 0.25
MLP_LR       = 3e-4
MLP_WD       = 1e-4
MLP_EPOCHS   = 250
MLP_PATIENCE = 40
MLP_BATCH    = 512
DEVICE       = "cpu"

# TFT hyperparams
TFT_D_MODEL  = 128
TFT_N_HEADS  = 4
TFT_N_LSTM   = 2
TFT_DROPOUT  = 0.15
TFT_EPOCHS   = 200
TFT_PATIENCE = 30
TFT_BATCH    = 32
QUANTILES    = (0.1, 0.5, 0.9)


# ===========================================================================
# Data loading helpers
# ===========================================================================

def load_may_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 18.05_test_dataset.csv -> (historical_with_power, may18_to_predict)."""
    df = pd.read_csv(path)
    # Round fractional-second timestamps to the nearest hour
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL]).dt.round("h")
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df["ws_180m_is_imputed"] = False

    hist = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    pred = df[df[TARGET_COL].isna()].copy().reset_index(drop=True)
    return hist, pred


def load_train_raw(path: Path) -> pd.DataFrame:
    """Load train CSV, sort ascending, add imputation placeholder."""
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df["ws_180m_is_imputed"] = False
    return df


# ===========================================================================
# Feature pipeline helpers (reused from train_best.py)
# ===========================================================================

def merge_era5(df: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    era5 = era5.copy().rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})
    df = df.merge(era5, on=TIMESTAMP_COL, how="left")
    df["ws10_bias"]     = df["wind_speed_10m"] - df["era5_wind_speed_10m"]
    df["ws100_vs_80"]   = df["era5_wind_speed_100m"] - df["wind_speed_80m"]
    df["gust_bias"]     = df["wind_gusts_10m"] - df["era5_wind_gusts_10m"]
    df["pressure_bias"] = df["pressure_msl"] - df["era5_pressure_msl"]
    df["era5_ws100_cube"]     = df["era5_wind_speed_100m"] ** 3
    df["era5_ws100_x_active"] = df["era5_wind_speed_100m"] ** 3 * df["active_turbines_ratio"]
    era5_dir_rad = np.deg2rad(df["era5_wind_direction_100m"])
    df["era5_dir100_sin"] = np.sin(era5_dir_rad)
    df["era5_dir100_cos"] = np.cos(era5_dir_rad)
    era5_new = [c for c in df.columns if c.startswith("era5_") or c.endswith("_bias")
                or c == "ws100_vs_80"]
    df[era5_new] = df[era5_new].fillna(0)
    return df


def add_nwp_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """NWP-based rolling features (mirror of ERA5 rolling for temporal context)."""
    df = df.copy()
    ws120  = df["wind_speed_120m"]
    pres   = df["pressure_msl"]
    dir120_sin = df["wind_dir_120m_sin"] if "wind_dir_120m_sin" in df.columns else pd.Series(0, index=df.index)
    dir120_cos = df["wind_dir_120m_cos"] if "wind_dir_120m_cos" in df.columns else pd.Series(0, index=df.index)
    for w in [3, 6, 12, 24]:
        roll = ws120.rolling(w, min_periods=1)
        df[f"nwp_ws120_roll_mean_{w}h"] = roll.mean()
        df[f"nwp_ws120_roll_std_{w}h"]  = roll.std().fillna(0)
    df["nwp_ws120_diff1"] = ws120.diff(1).fillna(0)
    df["nwp_ws120_diff3"] = ws120.diff(3).fillna(0)
    df["nwp_ws120_diff6"] = ws120.diff(6).fillna(0)
    df["nwp_pressure_diff3"]  = pres.diff(3).fillna(0)
    df["nwp_pressure_diff6"]  = pres.diff(6).fillna(0)
    df["nwp_pressure_diff12"] = pres.diff(12).fillna(0)
    roll6 = ws120.rolling(6, min_periods=1)
    df["nwp_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)
    df["nwp_dir_sin_diff1"] = dir120_sin.diff(1).fillna(0)
    df["nwp_dir_cos_diff1"] = dir120_cos.diff(1).fillna(0)
    return df


def add_era5_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ws   = df["era5_wind_speed_100m"]
    pres = df["era5_pressure_msl"]
    for w in [3, 6, 12, 24]:
        roll = ws.rolling(w, min_periods=1)
        df[f"era5_ws100_roll_mean_{w}h"] = roll.mean()
        df[f"era5_ws100_roll_std_{w}h"]  = roll.std().fillna(0)
    df["era5_ws100_diff1"] = ws.diff(1).fillna(0)
    df["era5_ws100_diff3"] = ws.diff(3).fillna(0)
    df["era5_ws100_diff6"] = ws.diff(6).fillna(0)
    df["era5_pressure_diff3"]  = pres.diff(3).fillna(0)
    df["era5_pressure_diff6"]  = pres.diff(6).fillna(0)
    df["era5_pressure_diff12"] = pres.diff(12).fillna(0)
    roll6 = ws.rolling(6, min_periods=1)
    df["era5_turb_intensity_6h"] = roll6.std().fillna(0) / (roll6.mean() + 1e-3)
    dir_sin = df["era5_dir100_sin"]
    dir_cos = df["era5_dir100_cos"]
    df["era5_dir_sin_diff1"] = dir_sin.diff(1).fillna(0)
    df["era5_dir_cos_diff1"] = dir_cos.diff(1).fillna(0)
    return df


def add_power_curve_features(df, pc_sector, pc_global):
    df = df.copy()
    v_eff    = df["v_eff"].to_numpy()
    dir_deg  = (df["wind_direction_120m"] * 1000.0).to_numpy()
    df["p_curve_sector"]          = pc_sector.predict(v_eff, dir_deg)
    df["p_curve_global"]          = pc_global.predict(v_eff)
    df["p_curve_rews"]            = pc_global.predict(df["rews"].to_numpy())
    df["p_curve_x_active"]        = df["p_curve_sector"] * df["active_turbines_ratio"]
    df["p_curve_global_x_active"] = df["p_curve_global"] * df["active_turbines_ratio"]
    df["p_curve_ratio"]           = df["p_curve_sector"] / CAPACITY_MW
    df["p_curve_sector_minus_global"] = df["p_curve_sector"] - df["p_curve_global"]
    return df


def to_cf(y_mw, active_turbines):
    p_avail = active_turbines * TURBINE_RATED_MW
    return y_mw / np.maximum(p_avail, 1e-3)


def from_cf(cf, active_turbines):
    p_avail = active_turbines * TURBINE_RATED_MW
    return cf * p_avail


# ===========================================================================
# LightGBM training
# ===========================================================================

def recency_weights(timestamps: np.ndarray) -> np.ndarray:
    """Per-row sample weights: recent years weighted more to reduce NWP drift."""
    years = pd.DatetimeIndex(timestamps).year
    w = np.array([_YEAR_WEIGHTS.get(int(y), _YEAR_WEIGHT_2026) for y in years],
                 dtype=np.float32)
    return w


def train_lgbm_ensemble(X_tr, y_tr, X_va, y_va, X_te, feat_cols, ws_tr, seeds, config,
                        sample_weights: np.ndarray | None = None):
    """3 regime specialists x N seeds -> average test prediction."""
    regime_test = {}
    base_w = sample_weights if sample_weights is not None else np.ones(len(y_tr), dtype=np.float32)
    for name, (lo, hi) in [("low", (0, 7)), ("mid", (4, 12)), ("high", (8, 25))]:
        mask    = (ws_tr >= lo) & (ws_tr < hi)
        weights = base_w * np.where(mask, 2.0, 0.3).astype(np.float32)
        tp_list = []
        for s in seeds:
            cfg = LGBMConfig(**{**config.__dict__, "seed": s})
            dt  = lgb.Dataset(X_tr, label=y_tr, weight=weights,
                              feature_name=feat_cols, free_raw_data=False)
            dv  = lgb.Dataset(X_va, label=y_va,
                              feature_name=feat_cols, free_raw_data=False)
            bst = lgb.train(
                cfg.to_params(), dt,
                num_boost_round=config.num_boost_round,
                valid_sets=[dv], valid_names=["val"],
                callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
            )
            tp_list.append(bst.predict(X_te, num_iteration=bst.best_iteration))
        regime_test[name] = np.mean(tp_list, axis=0)
    return np.mean(list(regime_test.values()), axis=0)


# ===========================================================================
# ResNet MLP (DL model -- inspired by CNN-LSTM literature, tabular adaptation)
# ===========================================================================

class ResBlock(nn.Module):
    """Residual block: Linear -> BN -> GELU -> Dropout -> Linear -> BN + skip."""
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class WindResNet(nn.Module):
    """Physics-aware residual MLP for wind power CF prediction.

    Sigmoid output: CF in [0, 1].
    Recommended by energies-18-00350 (LSTM MAPE 8.10%) and wind-05-00029-v2
    (hybrid DNN architectures for short-term WPF). Residual connections prevent
    vanishing gradients while keeping the model compact enough to avoid the
    overfitting that hurt MLP v108 on Q1 2026.
    """
    def __init__(self, n_in: int, hidden: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.res1 = ResBlock(hidden, dropout)
        self.res2 = ResBlock(hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.res2(self.res1(self.stem(x)))).squeeze(-1)


def train_mlp_single(X_tr, y_tr_cf, X_va, y_va_cf, seed: int,
                     ws_hub_tr=None, ws_hub_va=None):
    torch.manual_seed(seed); np.random.seed(seed)
    scaler = StandardScaler()
    col_means = np.nanmean(X_tr, axis=0)
    X_tr_f = np.where(np.isnan(X_tr), col_means, X_tr)
    X_va_f = np.where(np.isnan(X_va), col_means, X_va)
    X_tr_s = scaler.fit_transform(X_tr_f).astype(np.float32)
    X_va_s = scaler.transform(X_va_f).astype(np.float32)

    model = WindResNet(X_tr_s.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)

    X_t = torch.from_numpy(X_tr_s)
    y_t = torch.from_numpy(y_tr_cf.astype(np.float32))
    X_v = torch.from_numpy(X_va_s)
    y_v = torch.from_numpy(y_va_cf.astype(np.float32))

    use_physics = ws_hub_tr is not None
    if use_physics:
        physics_loss = PhysicsInformedLoss(quantiles=None, lambda_mono=0.08, lambda_cutin=0.05)
        ws_t = torch.from_numpy(ws_hub_tr.astype(np.float32))
        ws_v = torch.from_numpy(ws_hub_va.astype(np.float32))

    best_loss, best_state, patience = float("inf"), None, 0
    n = len(X_t)

    for _ in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, MLP_BATCH):
            idx = perm[start: start + MLP_BATCH]
            pred = model(X_t[idx])
            if use_physics:
                loss = physics_loss(pred, y_t[idx], ws_t[idx])
            else:
                loss = torch.mean(torch.abs(pred - y_t[idx]))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            if use_physics:
                val_loss = float(physics_loss(model(X_v), y_v, ws_v))
            else:
                val_loss = float(torch.mean(torch.abs(model(X_v) - y_v)))
        if val_loss < best_loss - 1e-5:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= MLP_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, scaler, col_means


def predict_mlp(model, scaler, col_means, X):
    X_f = np.where(np.isnan(X), col_means, X)
    X_s = scaler.transform(X_f).astype(np.float32)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_s)).numpy()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    set_global_seed(42)
    print("=" * 65)
    print("Predict May 18, 2026  |  LGBM(CF+MW) + ResNet MLP ensemble")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    df_train = load_train_raw(TRAIN_PATH)
    df_may_hist, df_may_pred = load_may_file(MAY_PATH)
    era5 = pd.read_parquet(ERA5_PATH) if USE_ERA5 else None

    print(f"  Train rows   : {len(df_train):,}")
    print(f"  May hist rows: {len(df_may_hist):,}")
    print(f"  Predict rows : {len(df_may_pred)} (May 18, 2026 hours 0-23)")
    print(f"  ERA5         : {'loaded' if USE_ERA5 else 'EXCLUDED (NWP-only)'}")

    # Preserve original row order of prediction set for output
    df_may_pred = df_may_pred.copy()
    df_may_pred["_orig_row"] = range(len(df_may_pred))

    # ------------------------------------------------------------------
    # 2. Build features on full combined timeline for rolling continuity
    # ------------------------------------------------------------------
    print("\n[2/5] Building features...")
    df_train["_split"]    = "train"
    df_may_hist["_split"] = "train"   # historical actuals -> augment training
    df_may_pred["_split"] = "test"

    combined = pd.concat([df_train, df_may_hist, df_may_pred],
                         ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)

    combined = build_features(combined, sort_by_time=False)
    if USE_ERA5:
        combined = merge_era5(combined, era5)
        combined = add_era5_rolling(combined)
    else:
        combined = add_nwp_rolling(combined)
        print("  ERA5 excluded: NWP rolling features added instead")
    combined = add_datasheet_power_features(combined)
    combined = add_extra_features(combined)

    df_all_train  = combined[combined["_split"] == "train"].reset_index(drop=True)
    df_test_feats = combined[combined["_split"] == "test"].reset_index(drop=True)

    impossible     = identify_impossible_rows(df_all_train)
    df_all_train["_is_impossible"] = impossible.values

    print(f"  All-train rows     : {len(df_all_train):,}")
    print(f"  Impossible rows    : {impossible.sum()}")
    print(f"  Test (predict) rows: {len(df_test_feats)}")

    # ------------------------------------------------------------------
    # 3. Feature selection probe using full 2025 as validation
    # ------------------------------------------------------------------
    print("\n[3/5] Feature selection (2025 full-year holdout as probe val)...")

    probe_train = df_all_train[df_all_train[TIMESTAMP_COL] < PROBE_VAL_START].copy()
    probe_val   = df_all_train[df_all_train[TIMESTAMP_COL] >= PROBE_VAL_START].copy()
    # Exclude Apr-May 2026 from probe_val (those go to LGBM training below)
    probe_val   = probe_val[probe_val[TIMESTAMP_COL] < pd.Timestamp("2026-01-01")].copy()
    print(f"  Probe train: {len(probe_train):,}  Probe val: {len(probe_val):,}")

    fit_probe = probe_train[~probe_train["_is_impossible"]]
    pc_s_p    = fit_sector_isotonic(fit_probe, n_sectors=8)
    pc_g_p    = IsotonicPowerCurve().fit(fit_probe["v_eff"], fit_probe[TARGET_COL])
    wk_p      = fit_wake_lookup(fit_probe, n_sectors=16)

    df_pt = add_power_curve_features(probe_train, pc_s_p, pc_g_p)
    df_pt = add_wake_features(df_pt, wk_p)
    df_pv = add_power_curve_features(probe_val, pc_s_p, pc_g_p)
    df_pv = add_wake_features(df_pv, wk_p)

    feat_cols_all = [c for c in feature_columns(df_pt)
                     if c not in ("_is_impossible", "_split", "_orig_row", TARGET_COL)]

    a_pt = df_pt["active_turbines"].to_numpy(dtype=np.float32)
    y_pt = to_cf(df_pt[TARGET_COL].to_numpy(dtype=np.float32), a_pt)
    a_pv = df_pv["active_turbines"].to_numpy(dtype=np.float32)
    y_pv = to_cf(df_pv[TARGET_COL].to_numpy(dtype=np.float32), a_pv)

    cfg0 = LGBMConfig(**{**LGBM_PARAMS.__dict__, "seed": 42})
    dt_  = lgb.Dataset(df_pt[feat_cols_all].to_numpy(dtype=np.float32),
                       label=y_pt, feature_name=feat_cols_all, free_raw_data=False)
    dv_  = lgb.Dataset(df_pv[feat_cols_all].to_numpy(dtype=np.float32),
                       label=y_pv, feature_name=feat_cols_all, free_raw_data=False)
    probe_bst = lgb.train(
        cfg0.to_params(), dt_, num_boost_round=5000,
        valid_sets=[dv_], valid_names=["val"],
        callbacks=[lgb.early_stopping(250, verbose=False)],
    )
    imp = probe_bst.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols_all, imp), key=lambda x: -x[1])
    top_k = [n for n, _ in feat_imp[:K]]

    probe_val_pred_cf  = probe_bst.predict(df_pv[feat_cols_all].to_numpy(dtype=np.float32))
    probe_val_pred_mw  = np.clip(from_cf(probe_val_pred_cf, a_pv), 0, CAPACITY_MW)
    probe_nmae = normalized_mae(df_pv[TARGET_COL].to_numpy(), probe_val_pred_mw)
    print(f"  Top-{K} features selected  (probe nMAE on Apr-May: {probe_nmae:.3f}%)")
    print(f"  Top-3 features: {[n for n, _ in feat_imp[:3]]}")

    # Fit final power curves + wake on ALL clean training data
    fit_all = df_all_train[~df_all_train["_is_impossible"]]
    pc_sector = fit_sector_isotonic(fit_all, n_sectors=8)
    pc_global = IsotonicPowerCurve().fit(fit_all["v_eff"], fit_all[TARGET_COL])
    wake      = fit_wake_lookup(fit_all, n_sectors=16)

    df_tr_final = add_power_curve_features(df_all_train, pc_sector, pc_global)
    df_tr_final = add_wake_features(df_tr_final, wake)
    df_te_final = add_power_curve_features(df_test_feats, pc_sector, pc_global)
    df_te_final = add_wake_features(df_te_final, wake)
    for c in set(top_k) - set(df_te_final.columns):
        df_te_final[c] = 0.0

    active_tr = df_tr_final["active_turbines"].to_numpy(dtype=np.float32)
    y_tr_mw   = df_tr_final[TARGET_COL].to_numpy(dtype=np.float32)
    y_tr_cf   = to_cf(y_tr_mw, active_tr)
    active_te = df_te_final["active_turbines"].to_numpy(dtype=np.float32)
    ws_tr     = df_tr_final["wind_speed_120m"].to_numpy()

    X_tr = df_tr_final[top_k].to_numpy(dtype=np.float32)
    X_te = df_te_final[top_k].to_numpy(dtype=np.float32)

    # Seasonal filter: keep only March-May rows from all years.
    # ES val: March-May 2025 (same season as test day, most recent spring before 2026).
    # Apr-May 2026 historical data goes into TRAINING so the model learns 2026 NWP bias.
    ts_tr = df_tr_final[TIMESTAMP_COL]
    spring_mask = ts_tr.dt.month.isin(SPRING_MONTHS)
    val_mask    = spring_mask & (ts_tr >= SPRING_ES_START) & (ts_tr < SPRING_ES_END)
    tr_mask     = spring_mask & ~val_mask

    X_va_lgbm  = X_tr[val_mask]
    y_va_cf    = y_tr_cf[val_mask]
    X_tr_lgbm  = X_tr[tr_mask]
    y_tr_cf_es = y_tr_cf[tr_mask]
    y_tr_mw_es = y_tr_mw[tr_mask]
    ws_tr_es   = ws_tr[tr_mask]

    # Recency weights: up-weight 2026 data so LGBM learns 2026 NWP bias pattern
    ts_lgbm_train = ts_tr[tr_mask]
    sw_tr = recency_weights(ts_lgbm_train.to_numpy())
    n_2026 = (ts_lgbm_train >= pd.Timestamp("2026-01-01")).sum()
    print(f"  Seasonal filter: Mar-May only")
    print(f"  LGBM train: {tr_mask.sum():,} rows  "
          f"(ES val: {val_mask.sum()} rows, Mar-May 2025)")
    print(f"  2026 rows in training: {n_2026}  (weight {_YEAR_WEIGHT_2026}x)")

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    print("\n[4/5] Training ensembles...")

    # --- LGBM CF target ---
    print("  [LGBM CF]")
    test_cf = train_lgbm_ensemble(
        X_tr_lgbm, y_tr_cf_es, X_va_lgbm, y_va_cf,
        X_te, top_k, ws_tr_es, SEEDS, LGBM_PARAMS, sample_weights=sw_tr,
    )

    # --- LGBM MW target ---
    print("  [LGBM MW]")
    y_tr_mw_es_f = y_tr_mw_es.astype(np.float32)
    y_va_mw_lgbm = y_tr_mw[val_mask]
    test_mw_raw = train_lgbm_ensemble(
        X_tr_lgbm, y_tr_mw_es_f, X_va_lgbm, y_va_mw_lgbm,
        X_te, top_k, ws_tr_es, SEEDS, LGBM_PARAMS, sample_weights=sw_tr,
    )

    # --- ResNet MLP CF target (3 seeds, physics-informed loss) ---
    # HYBRID: MLP uses FULL training data (not spring-only).
    # Spring filter hurts MLP (10.7% vs 7.7%) because 6,954 rows is too few
    # for 80 features + residual blocks.
    print("  [ResNet MLP CF + physics loss]  (FULL DATA — no spring filter)")
    ws_hub_all = df_tr_final["wind_speed_80m"].to_numpy(dtype=np.float32)
    # Use all non-impossible rows for MLP training
    impossible_mask = df_tr_final["_is_impossible"].to_numpy()
    mlp_tr_mask = ~impossible_mask
    mlp_va_mask = val_mask  # keep the same spring val for ES monitoring
    X_tr_mlp    = X_tr[mlp_tr_mask]
    y_tr_cf_mlp = y_tr_cf[mlp_tr_mask]
    X_va_mlp    = X_tr[mlp_va_mask]
    y_va_cf_mlp = y_tr_cf[mlp_va_mask]
    ws_hub_tr_arr = ws_hub_all[mlp_tr_mask]
    ws_hub_va_arr = ws_hub_all[mlp_va_mask]
    mlp_test_preds = []
    for s in SEEDS:
        model_s, scaler_s, means_s = train_mlp_single(
            X_tr_mlp, y_tr_cf_mlp, X_va_mlp, y_va_cf_mlp, seed=s,
            ws_hub_tr=ws_hub_tr_arr, ws_hub_va=ws_hub_va_arr,
        )
        te_cf = predict_mlp(model_s, scaler_s, means_s, X_te)
        mlp_test_preds.append(te_cf)
        va_cf_hat = predict_mlp(model_s, scaler_s, means_s, X_va_mlp)
        va_mw_hat = np.clip(from_cf(va_cf_hat, df_tr_final["active_turbines"].to_numpy(dtype=np.float32)[mlp_va_mask]), 0, CAPACITY_MW)
        y_va_mw_mlp = y_tr_mw[mlp_va_mask]
        print(f"    Seed {s}: MLP nMAE(spring-2025 ES val) = {normalized_mae(y_va_mw_mlp, va_mw_hat):.3f}%")
    test_mlp_cf = np.mean(mlp_test_preds, axis=0)

    # --- WindTFT: 24h sequence model (3 seeds) ---
    print("  [WindTFT sequence model]")
    df_tft_src = df_tr_final[top_k + [TIMESTAMP_COL]].copy()
    df_tft_src["_cf"]  = y_tr_cf
    df_tft_src["_ws80"] = df_tr_final["wind_speed_80m"].fillna(0.0).values
    # Seasonal filter: same spring-months restriction as LGBM/MLP
    df_tft_src = df_tft_src[df_tft_src[TIMESTAMP_COL].dt.month.isin(SPRING_MONTHS)].reset_index(drop=True)
    X_days, y_days, ws_days = make_day_sequences(
        df_tft_src, top_k, "_cf", "_ws80", TIMESTAMP_COL,
    )
    print(f"    Complete spring training days: {len(X_days)}")
    # Use 20 val days (Mar-May 2025 season) so 2026 days stay in TFT training
    n_val_days  = min(20, len(X_days) // 10)
    X_tr_d, y_tr_d, ws_tr_d = X_days[:-n_val_days], y_days[:-n_val_days], ws_days[:-n_val_days]
    X_va_d, y_va_d, ws_va_d = X_days[-n_val_days:], y_days[-n_val_days:], ws_days[-n_val_days:]

    df_te_sorted = df_te_final.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    X_test_day  = df_te_sorted[top_k].to_numpy(dtype=np.float32)   # [24, n_feat]

    tft_preds_q = []
    for s in SEEDS:
        tft, tft_sc, tft_means = train_tft(
            X_tr_d, y_tr_d, ws_tr_d, X_va_d, y_va_d, ws_va_d,
            quantiles=QUANTILES,
            d_model=TFT_D_MODEL, n_heads=TFT_N_HEADS, n_lstm=TFT_N_LSTM,
            dropout=TFT_DROPOUT, epochs=TFT_EPOCHS, patience=TFT_PATIENCE,
            batch_size=TFT_BATCH, device=DEVICE, seed=s,
        )
        pred_q = predict_tft_day(tft, tft_sc, tft_means, X_test_day, device=DEVICE)
        tft_preds_q.append(pred_q)  # [24, 3]
        # Val nMAE on last val-day batch (approximate)
        va_p50 = np.array([
            predict_tft_day(tft, tft_sc, tft_means, X_va_d[d], device=DEVICE)[:, 1]
            for d in range(len(X_va_d))
        ])  # [n_val_days, 24]
        va_cf_flat = va_p50.reshape(-1)
        # active_turbines for val days -- use mean active from training
        active_mean = float(df_tr_final["active_turbines"].mean())
        va_mw_flat  = np.clip(va_cf_flat * active_mean * TURBINE_RATED_MW, 0, CAPACITY_MW)
        va_tgt_flat = y_va_d.reshape(-1) * active_mean * TURBINE_RATED_MW
        print(f"    Seed {s}: TFT nMAE(ES-val-{n_val_days}d) = "
              f"{normalized_mae(va_tgt_flat, va_mw_flat):.3f}%")

    tft_avg_q   = np.mean(tft_preds_q, axis=0)          # [24, 3] — P10/P50/P90 averaged
    tft_p10_cf  = tft_avg_q[:, 0]
    tft_p50_cf  = tft_avg_q[:, 1]
    tft_p90_cf  = tft_avg_q[:, 2]

    # Align TFT output to original row order of df_te_final
    te_hour_order = df_te_sorted["_orig_row"].to_numpy().astype(int)
    tft_p10_ordered = np.empty(24); tft_p50_ordered = np.empty(24); tft_p90_ordered = np.empty(24)
    tft_p10_ordered[te_hour_order] = tft_p10_cf
    tft_p50_ordered[te_hour_order] = tft_p50_cf
    tft_p90_ordered[te_hour_order] = tft_p90_cf

    # ------------------------------------------------------------------
    # 5. Blend and output
    # ------------------------------------------------------------------
    print("\n[5/5] Blending and writing output...")

    pred_cf_mw  = np.clip(from_cf(test_cf,    active_te), 0, CAPACITY_MW)
    pred_mw_mw  = np.clip(test_mw_raw,                    0, CAPACITY_MW)
    pred_mlp_mw = np.clip(from_cf(test_mlp_cf, active_te), 0, CAPACITY_MW)
    # TFT P50: CF -> MW using active turbines (sorted to match active_te order)
    pred_tft_mw = np.clip(from_cf(tft_p50_ordered, active_te), 0, CAPACITY_MW)
    pred_p10_mw = np.clip(from_cf(tft_p10_ordered, active_te), 0, CAPACITY_MW)
    pred_p90_mw = np.clip(from_cf(tft_p90_ordered, active_te), 0, CAPACITY_MW)

    final_mw = (BLEND_CF  * pred_cf_mw
              + BLEND_MW  * pred_mw_mw
              + BLEND_MLP * pred_mlp_mw
              + BLEND_TFT * pred_tft_mw)
    final_mw = np.clip(final_mw, 0, CAPACITY_MW)

    # Restore original row order (descending, as in source file)
    orig_order = df_te_final["_orig_row"].to_numpy().astype(int)
    out_mw = np.empty_like(final_mw)
    out_mw[orig_order] = final_mw

    out_ts = np.empty(len(df_te_final), dtype=object)
    out_ts[orig_order] = df_te_final[TIMESTAMP_COL].to_numpy()

    out_p10 = np.empty_like(final_mw); out_p10[orig_order] = pred_p10_mw
    out_p90 = np.empty_like(final_mw); out_p90[orig_order] = pred_p90_mw
    out_df = pd.DataFrame({
        "datetime":    pd.to_datetime(out_ts).strftime("%Y-%m-%d %H:%M:%S"),
        "hour":        df_may_pred["hour_of_day"].to_numpy(),
        "forecast_mw": out_mw,
        "p10_mw":      out_p10,
        "p90_mw":      out_p90,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'=' * 65}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Predictions (24 hours, May 18, 2026):")
    print(f"{'Hour':>6}  {'Forecast MW':>12}  {'LGBM_CF':>10}  {'LGBM_MW':>10}  {'MLP':>10}")
    print(f"  {'-'*55}")
    # Sort by hour for display
    sort_idx = np.argsort(out_df["hour"].to_numpy())
    for i in sort_idx:
        h   = int(out_df.iloc[i]["hour"])
        mw  = float(out_df.iloc[i]["forecast_mw"])
        cf_ = float(pred_cf_mw[orig_order[i]]) if orig_order[i] < len(pred_cf_mw) else 0
        mwv = float(pred_mw_mw[orig_order[i]]) if orig_order[i] < len(pred_mw_mw) else 0
        ml_ = float(pred_mlp_mw[orig_order[i]]) if orig_order[i] < len(pred_mlp_mw) else 0
        print(f"  {h:>4}h  {mw:>12.3f}  {cf_:>10.3f}  {mwv:>10.3f}  {ml_:>10.3f}")
    print(f"  {'-'*55}")
    print(f"  Mean  : {out_mw.mean():.3f} MW")
    print(f"  Std   : {out_mw.std():.3f} MW")
    print(f"  Range : [{out_mw.min():.3f}, {out_mw.max():.3f}] MW")
    print(f"  Active turbines: {int(active_te.mean())} / {TOTAL_TURBINES}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
