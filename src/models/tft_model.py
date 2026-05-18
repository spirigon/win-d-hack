"""
Temporal Fusion Transformer for 24-hour direct wind power forecasting,
with physics-informed training via monotonicity and cut-in constraints.

Architecture: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", Int. J. Forecasting 37(4).

Physics constraints
-------------------
1. Monotonicity  — P(v_hi) >= P(v_lo) when v_hi > v_lo  (power curve is non-decreasing)
2. Cut-in        — P ~ 0 when hub-height wind < CUT_IN_MS
3. Sorted quantile output  — guarantees P10 <= P50 <= P90 at inference
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Physics-Informed Loss
# ---------------------------------------------------------------------------

class PhysicsInformedLoss(nn.Module):
    """Combined prediction + physics constraint loss.

    Supports two shapes:
      - Point mode     pred_cf [B]        target [B]     ws [B]   -- ResNet MLP
      - Sequence mode  pred_cf [B, T, Q]  target [B, T]  ws [B, T] -- WindTFT

    total = main_loss
          + lambda_mono  * monotonicity_penalty
          + lambda_cutin * cut_in_penalty
    """

    CUT_IN_MS    = 3.0    # hub-height cut-in (m/s)
    CUTIN_MAX_CF = 0.02   # allow up to 2 % CF below cut-in (upper-rotor residual)
    MONO_GAP_MS  = 0.5    # min ws gap to trigger monotonicity enforcement

    def __init__(
        self,
        quantiles:    tuple[float, ...] | None = None,
        lambda_mono:  float = 0.08,
        lambda_cutin: float = 0.05,
    ) -> None:
        super().__init__()
        self.quantiles    = quantiles
        self.lambda_mono  = lambda_mono
        self.lambda_cutin = lambda_cutin
        self._p50_idx = (
            min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
            if quantiles else 0
        )

    def forward(
        self,
        pred_cf:   torch.Tensor,
        target_cf: torch.Tensor,
        ws_hub:    torch.Tensor,
    ) -> torch.Tensor:
        seq_mode = pred_cf.ndim == 3

        # main loss
        if self.quantiles is None:
            p = pred_cf.squeeze(-1) if pred_cf.ndim == 2 else pred_cf
            main = F.l1_loss(p, target_cf)
        else:
            main = self._quantile_loss(pred_cf, target_cf)

        # P50 for physics constraints
        if pred_cf.ndim == 1:
            p50 = pred_cf
        elif pred_cf.ndim == 2:
            p50 = pred_cf[:, self._p50_idx]
        else:
            p50 = pred_cf[:, :, self._p50_idx]  # [B, T]

        mono  = self._seq_mono(p50, ws_hub) if seq_mode else self._batch_mono(p50, ws_hub)
        cutin = self._cutin(p50.reshape(-1), ws_hub.reshape(-1))

        return main + self.lambda_mono * mono + self.lambda_cutin * cutin

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_zeros(())
        for i, q in enumerate(self.quantiles):
            e = target - pred[..., i]
            total = total + torch.mean(torch.max(q * e, (q - 1.0) * e))
        return total / len(self.quantiles)

    def _batch_mono(self, pred: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        if len(pred) < 2:
            return pred.new_zeros(())
        idx    = torch.argsort(ws)
        ws_s   = ws[idx];  pred_s = pred[idx]
        gap    = ws_s[1:] - ws_s[:-1]
        diff   = pred_s[1:] - pred_s[:-1]
        return (F.relu(-diff) * (gap > self.MONO_GAP_MS).float()).mean()

    def _seq_mono(self, pred: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        # Compare every pair of time-steps within each day: O(B * T^2)
        wi = ws.unsqueeze(2);   wj = ws.unsqueeze(1)    # [B, T, 1] / [B, 1, T]
        pi = pred.unsqueeze(2); pj = pred.unsqueeze(1)
        sig = ((wi - wj) > self.MONO_GAP_MS).float()    # where wi is windier
        return (F.relu(-(pi - pj)) * sig).mean()

    def _cutin(self, pred: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        below = (ws < self.CUT_IN_MS).float()
        return (F.relu(pred - self.CUTIN_MAX_CF) * below).mean()


# ---------------------------------------------------------------------------
# 2. TFT building blocks
# ---------------------------------------------------------------------------

class GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021 §3.3)."""

    def __init__(self, d_in: int, d_hid: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_in, d_hid)
        self.fc2  = nn.Linear(d_hid, d_out)   # value
        self.fc3  = nn.Linear(d_hid, d_out)   # gate
        self.skip = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h     = self.drop(F.elu(self.fc1(x)))
        gate  = torch.sigmoid(self.fc3(h))
        return self.norm(self.skip(x) + self.fc2(h) * gate)


class VSN(nn.Module):
    """Variable Selection Network: soft per-feature importance weights."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.selector = GRN(n_vars, d_model, n_vars, dropout)
        self.proj     = nn.Linear(n_vars, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.selector(x), dim=-1)
        return self.norm(self.proj(x * weights))


# ---------------------------------------------------------------------------
# 3. WindTFT model
# ---------------------------------------------------------------------------

class WindTFT(nn.Module):
    """TFT for 24-hour direct wind power forecasting.

    Input  : [B, 24, n_features]   — full-day NWP profile (all future covariates known)
    Output : [B, 24, n_quantiles]  — CF in (0,1), sorted so P10 <= P50 <= P90

    Pipeline
    --------
    VSN -> LSTM (+ residual) -> Multi-head Self-Attention (+ residual) -> GRN -> head
    """

    def __init__(
        self,
        n_features:  int,
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_lstm:      int   = 2,
        n_quantiles: int   = 3,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.vsn       = VSN(n_features, d_model, dropout)
        self.lstm      = nn.LSTM(
            d_model, d_model, n_lstm,
            batch_first=True,
            dropout=dropout if n_lstm > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.out_grn   = GRN(d_model, d_model, d_model, dropout)
        self.head      = nn.Linear(d_model, n_quantiles)
        self.n_q       = n_quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.vsn(x)                           # [B, 24, d]
        lstm_out, _ = self.lstm(h)
        h = self.lstm_norm(h + lstm_out)          # residual
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)          # residual
        h = self.out_grn(h)
        out = torch.sigmoid(self.head(h))         # [B, 24, n_q]
        if self.n_q > 1:
            out = torch.sort(out, dim=-1).values  # enforce P10 <= P50 <= P90
        return out


# ---------------------------------------------------------------------------
# 4. Dataset and data preparation
# ---------------------------------------------------------------------------

class DayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ws: np.ndarray) -> None:
        self.X  = torch.from_numpy(X.astype(np.float32))
        self.y  = torch.from_numpy(y.astype(np.float32))
        self.ws = torch.from_numpy(ws.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.ws[idx]


def make_day_sequences(
    df:            pd.DataFrame,
    feature_cols:  list[str],
    target_col:    str,
    ws_col:        str,
    timestamp_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape hourly DataFrame into complete 24-step daily arrays.

    Incomplete days and days with any NaN target are silently dropped.

    Returns
    -------
    X_days  : float32 [n_days, 24, n_features]
    y_days  : float32 [n_days, 24]
    ws_days : float32 [n_days, 24]
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df[timestamp_col]).dt.date

    X_list, y_list, ws_list = [], [], []
    for _, grp in df.groupby("_date", sort=True):
        if len(grp) != 24:
            continue
        grp = grp.sort_values(timestamp_col)
        if grp[target_col].isna().any():
            continue
        X_list.append(grp[feature_cols].to_numpy(dtype=np.float32))
        y_list.append(grp[target_col].to_numpy(dtype=np.float32))
        ws_list.append(grp[ws_col].to_numpy(dtype=np.float32))

    if not X_list:
        raise ValueError("No complete 24-hour days found in the dataset.")

    return np.stack(X_list), np.stack(y_list), np.stack(ws_list)


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

def train_tft(
    X_tr:  np.ndarray,
    y_tr:  np.ndarray,
    ws_tr: np.ndarray,
    X_va:  np.ndarray,
    y_va:  np.ndarray,
    ws_va: np.ndarray,
    *,
    quantiles:    tuple[float, ...] = (0.1, 0.5, 0.9),
    d_model:      int   = 128,
    n_heads:      int   = 4,
    n_lstm:       int   = 2,
    dropout:      float = 0.15,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    epochs:       int   = 200,
    patience:     int   = 30,
    batch_size:   int   = 32,
    lambda_mono:  float = 0.08,
    lambda_cutin: float = 0.05,
    device:       str   = "cpu",
    seed:         int   = 42,
) -> tuple[WindTFT, StandardScaler, np.ndarray]:
    """Train WindTFT on daily sequences. Returns (model, scaler, col_means)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tr, seq_len, n_feat = X_tr.shape
    col_means = np.nanmean(X_tr.reshape(-1, n_feat), axis=0)

    def _scale(X: np.ndarray, scaler=None) -> tuple[np.ndarray, StandardScaler]:
        n, t, f = X.shape
        Xf = np.where(np.isnan(X), col_means, X).reshape(-1, f)
        if scaler is None:
            scaler = StandardScaler()
            return scaler.fit_transform(Xf).reshape(n, t, f).astype(np.float32), scaler
        return scaler.transform(Xf).reshape(n, t, f).astype(np.float32), scaler

    X_tr_s, scaler = _scale(X_tr)
    X_va_s, _      = _scale(X_va, scaler)

    model   = WindTFT(n_feat, d_model, n_heads, n_lstm, len(quantiles), dropout).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = PhysicsInformedLoss(quantiles, lambda_mono, lambda_cutin)

    loader  = DataLoader(
        DayDataset(X_tr_s, y_tr, ws_tr),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    Xv = torch.from_numpy(X_va_s).to(device)
    yv = torch.from_numpy(y_va.astype(np.float32)).to(device)
    wv = torch.from_numpy(ws_va.astype(np.float32)).to(device)

    best_loss, best_state, no_imp = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb, wb in loader:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            loss = loss_fn(model(Xb), yb, wb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv, wv))

        if val_loss < best_loss - 1e-5:
            best_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, col_means


# ---------------------------------------------------------------------------
# 6. Inference
# ---------------------------------------------------------------------------

def predict_tft_day(
    model:     WindTFT,
    scaler:    StandardScaler,
    col_means: np.ndarray,
    X_day:     np.ndarray,   # [24, n_features]
    device:    str = "cpu",
) -> np.ndarray:
    """Predict CF quantiles for a single 24-hour day. Returns [24, n_quantiles]."""
    n_feat = X_day.shape[-1]
    X = X_day[np.newaxis]                              # [1, 24, n_feat]
    Xf = np.where(np.isnan(X), col_means, X).reshape(-1, n_feat)
    Xs = scaler.transform(Xf).reshape(1, 24, n_feat).astype(np.float32)

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xs).to(device))  # [1, 24, n_q]
    return out.squeeze(0).cpu().numpy()                # [24, n_q]
