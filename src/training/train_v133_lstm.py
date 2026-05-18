"""V133: Simple LSTM on weather sequences for temporal diversity.

The key insight: LGBM predicts each hour independently. An LSTM sees
6-12h of weather CONTEXT before each prediction, capturing:
- Wind ramp trajectories (accelerating vs decelerating)
- Pressure tendency (frontal passages)
- Temperature gradient evolution (boundary layer transitions)

Even a small LSTM with r=0.90-0.95 correlation to LGBM can provide
genuine diversity when blended at 5-10%.

Architecture:
  - Input: 12h lookback × N weather features per hour
  - 1-layer LSTM hidden=64
  - Linear head → capacity factor prediction
  - MAE loss, Adam optimizer
  - 5-fold walk-forward training (same folds as LGBM)
  - 5 seeds averaged

Target: capacity factor (same as v97b CF model)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TIMESTAMP_COL
from src.eval.metrics import normalized_mae
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
V131_BEST_PATH = _ROOT / "submissions" / "archive" / "v131.v128_s52.csv"  # LB=7.315
OUTPUT_DIR = _ROOT / "submissions" / "archive"

TCN = "Выработка. Результирующий расчет"
TOTAL_TURBINES = 26
MAINT_COL = "Кол-во_ВЭУ_в_ремонте"

# LSTM params
LOOKBACK = 12  # hours of context
HIDDEN = 64
N_LAYERS = 1
DROPOUT = 0.1
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 15
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2026]


class WindLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output in [0, 1] = capacity factor
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _build_weather_features(df: pd.DataFrame) -> np.ndarray:
    """Build per-hour weather feature vector."""
    ts = pd.to_datetime(df.iloc[:, 0])
    hour = ts.dt.hour.to_numpy(dtype=np.float32)
    month = ts.dt.month.to_numpy(dtype=np.float32)

    # Direction encoding
    dir_120 = df["wind_direction_120m"].to_numpy(dtype=np.float32)
    dir_rad = np.deg2rad(dir_120 * 1000.0)

    features = np.column_stack([
        df["wind_speed_120m"].to_numpy(dtype=np.float32),
        df["wind_speed_80m"].to_numpy(dtype=np.float32),
        df["wind_speed_10m"].to_numpy(dtype=np.float32),
        np.sin(dir_rad),
        np.cos(dir_rad),
        df["wind_gusts_10m"].to_numpy(dtype=np.float32),
        df["temperature_80m"].to_numpy(dtype=np.float32),
        df["pressure_msl"].to_numpy(dtype=np.float32) / 1000.0,  # scale
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
    ]).astype(np.float32)
    # Fill any remaining NaN with 0
    features = np.nan_to_num(features, nan=0.0)
    return features


def _create_sequences(features: np.ndarray, targets: np.ndarray | None,
                      lookback: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Create sliding window sequences. Returns (X, y, valid_indices)."""
    n = len(features)
    X_list = []
    y_list = []
    idx_list = []

    for i in range(lookback, n):
        X_list.append(features[i - lookback:i])
        if targets is not None:
            y_list.append(targets[i])
        idx_list.append(i)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if targets is not None else None
    indices = np.array(idx_list)
    return X, y, indices


def _normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize per-feature across the sequence dimension."""
    # X shape: (n_samples, lookback, n_features)
    # Compute mean/std across samples and time steps
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def _train_lstm(X_train, y_train, X_val, y_val, seed, input_dim):
    """Train single LSTM with early stopping."""
    set_global_seed(seed)

    model = WindLSTM(input_dim, HIDDEN, N_LAYERS, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train)
    X_v = torch.from_numpy(X_val)
    y_v = torch.from_numpy(y_val)

    train_ds = TensorDataset(X_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_mae = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = torch.abs(pred - yb).mean()
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_mae = torch.abs(val_pred - y_v).mean().item()

        if np.isnan(val_mae):
            continue

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_mae


def _write(preds, path, label=""):
    preds = np.clip(preds, 0.0, CAPACITY_MW)
    pd.DataFrame({TCN: preds}).to_csv(path, index=False, float_format="%.6f")
    print(f"  {path.name:<55s} mean={preds.mean():.2f}  {label}")


def main():
    set_global_seed(42)
    t_start = time.time()

    print("=" * 72)
    print(f"V133: LSTM on weather sequences (lookback={LOOKBACK}h, {N_SEEDS} seeds)")
    print("=" * 72)

    # --- Load and prepare data ---
    print("\n[1/3] Loading data and building sequences...")
    train = pd.read_csv(TRAIN_PATH)
    train.columns = train.columns.str.strip()
    valid = pd.read_csv(VALID_PATH)
    valid.columns = valid.columns.str.strip()

    # Concatenate train + valid for continuous sequence
    train["_split"] = "train"
    valid["_split"] = "valid"
    # Valid has no target column — add NaN placeholder
    valid[TCN] = np.nan
    combined = pd.concat([train, valid], ignore_index=True)
    combined["_ts"] = pd.to_datetime(combined.iloc[:, 0])
    combined = combined.sort_values("_ts").reset_index(drop=True)

    # Build features for full sequence
    features = _build_weather_features(combined)
    print(f"  Features per hour: {features.shape[1]}")
    print(f"  Total hours: {len(combined)} (train={len(train)}, valid={len(valid)})")

    # Target: capacity factor
    power = combined[TCN].to_numpy(dtype=np.float32)
    maint = combined[MAINT_COL].to_numpy(dtype=np.float32)
    active = TOTAL_TURBINES - maint
    cf_target = power / (active * (CAPACITY_MW / TOTAL_TURBINES) + 1e-6)
    cf_target = np.clip(cf_target, 0, 1)

    # Mark valid rows
    is_train = combined["_split"] == "train"
    is_valid = combined["_split"] == "valid"
    has_target = np.isfinite(power) & is_train.to_numpy()

    # Create sequences for the full timeline
    X_seq, _, seq_indices = _create_sequences(features, cf_target, LOOKBACK)
    # seq_indices[i] is the index in combined that sequence i predicts
    print(f"  Sequences created: {len(X_seq)}")

    # Split: train sequences (has target), valid sequences (for prediction)
    train_seq_mask = has_target[seq_indices]
    valid_seq_mask = is_valid.to_numpy()[seq_indices]

    X_train_all = X_seq[train_seq_mask]
    y_train_all = cf_target[seq_indices[train_seq_mask]]
    X_valid_all = X_seq[valid_seq_mask]
    valid_combined_idx = seq_indices[valid_seq_mask]

    # Remove impossible rows from training
    ws_at_target = features[seq_indices[train_seq_mask], 0]  # ws_120 (first feature)
    power_at_target = power[seq_indices[train_seq_mask]]
    impossible = (ws_at_target > 8) & (power_at_target < 5)
    good_mask = ~impossible & np.isfinite(y_train_all)
    X_train_all = X_train_all[good_mask]
    y_train_all = y_train_all[good_mask]

    print(f"  Train sequences: {len(X_train_all)}")
    print(f"  Valid sequences: {len(X_valid_all)}")

    # --- Walk-forward training ---
    print(f"\n[2/3] Training LSTM (walk-forward, {N_SEEDS} seeds)...")

    # Use timestamp-based split: train on 2023-Sep2024, validate on Q4 2024
    train_ts = combined["_ts"].to_numpy()
    train_seq_ts = train_ts[seq_indices[train_seq_mask][good_mask]]

    val_start = pd.Timestamp("2024-10-01")
    tr_mask = train_seq_ts < val_start
    va_mask = train_seq_ts >= val_start

    X_tr = X_train_all[tr_mask]
    y_tr = y_train_all[tr_mask]
    X_va = X_train_all[va_mask]
    y_va = y_train_all[va_mask]

    print(f"  Train: {len(X_tr)}, Val: {len(X_va)}")

    # Normalize
    X_tr_n, X_va_n = _normalize_features(X_tr, X_va)
    _, X_valid_n = _normalize_features(X_tr, X_valid_all)

    input_dim = X_tr.shape[-1]

    # Train multiple seeds
    val_preds_all = []
    test_preds_all = []

    for seed in SEEDS:
        t0 = time.time()
        model, best_mae = _train_lstm(X_tr_n, y_tr, X_va_n, y_va, seed, input_dim)

        with torch.no_grad():
            vp = model(torch.from_numpy(X_va_n)).numpy()
            tp = model(torch.from_numpy(X_valid_n)).numpy()

        val_preds_all.append(vp)
        test_preds_all.append(tp)
        print(f"    Seed {seed}: val_mae={best_mae:.4f}  ({time.time()-t0:.0f}s)")

    # Average predictions
    avg_val = np.mean(val_preds_all, axis=0)
    avg_test_cf = np.mean(test_preds_all, axis=0)

    # Convert CF to MW
    active_valid = active[valid_combined_idx]
    avg_test_mw = np.clip(avg_test_cf * active_valid * (CAPACITY_MW / TOTAL_TURBINES), 0, CAPACITY_MW)

    # Evaluate on validation period
    active_va = active[seq_indices[train_seq_mask][good_mask][va_mask]]
    val_mw = np.clip(avg_val * active_va * (CAPACITY_MW / TOTAL_TURBINES), 0, CAPACITY_MW)
    val_target_mw = power[seq_indices[train_seq_mask][good_mask][va_mask]]
    val_nmae = float(np.mean(np.abs(val_target_mw - val_mw)) / CAPACITY_MW * 100)
    print(f"\n  LSTM validation nMAE: {val_nmae:.4f}%")

    # --- Map to submission order ---
    print(f"\n[3/3] Building submission...")

    # valid_combined_idx gives the position in 'combined' for each prediction
    # We need to map these to the submission row order
    valid_ts_in_combined = combined["_ts"].iloc[valid_combined_idx].to_numpy()

    # Load the valid_features ordering (submission order)
    valid_raw = pd.read_csv(VALID_PATH)
    valid_raw.columns = valid_raw.columns.str.strip()
    sub_ts = pd.to_datetime(valid_raw.iloc[:, 0]).to_numpy()

    # Create timestamp → prediction map
    ts_to_pred = dict(zip(valid_ts_in_combined, avg_test_mw))
    lstm_preds = np.array([ts_to_pred.get(t, np.nan) for t in sub_ts], dtype=np.float64)

    # Fill any missing (first LOOKBACK hours won't have predictions)
    n_missing = np.isnan(lstm_preds).sum()
    if n_missing > 0:
        # Fill with mean of available predictions
        fill_val = np.nanmean(lstm_preds)
        lstm_preds[np.isnan(lstm_preds)] = fill_val
        print(f"  Filled {n_missing} missing predictions with mean={fill_val:.2f}")

    lstm_preds = np.clip(lstm_preds, 0, CAPACITY_MW)
    print(f"  LSTM predictions: mean={lstm_preds.mean():.2f}, std={lstm_preds.std():.2f}")

    # Correlation with v97b
    v131_best = pd.read_csv(V131_BEST_PATH)[TCN].values
    valid_both = np.isfinite(lstm_preds) & np.isfinite(v131_best)
    corr = np.corrcoef(lstm_preds[valid_both], v131_best[valid_both])[0, 1]
    print(f"  Correlation LSTM vs LB-best: {corr:.4f}")

    # Write standalone
    _write(lstm_preds, OUTPUT_DIR / "v133.0_lstm_standalone.csv", "LSTM standalone")

    # Blend with current LB best
    for w in [0.03, 0.05, 0.07, 0.10, 0.15]:
        blend = np.clip(w * lstm_preds + (1 - w) * v131_best, 0, CAPACITY_MW)
        _write(blend, OUTPUT_DIR / f"v133.1_lstm{int(w*100):02d}_best{int((1-w)*100):02d}.csv",
               f"{w:.0%} LSTM + {1-w:.0%} LB-best")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"\n  If LSTM corr < 0.97 with LB-best, blend at 5-10% should help")


if __name__ == "__main__":
    main()
