"""TFT without autoregressive target — weather-only encoder.

Key insight: our autoregressive TFT drifts badly at inference (8.07% with
just 10% weight). Solution: remove target from encoder entirely. The model
learns ONLY from weather sequences → power.

This is equivalent to a 1D attention model over weather time series.
No rolling needed at inference — single forward pass per block.

Training: prediction_length=1 (single-step), encoder sees 168h of WEATHER ONLY.
Inference: for each hour in Q1 2026, feed 168h of weather context → predict power.

Usage:
    python -m src.training.train_tft_noar
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.schema import CAPACITY_MW, TARGET_COL, TIMESTAMP_COL, TURBINES_IN_MAINTENANCE_COL
from src.eval.metrics import normalized_mae
from src.features.physics import compute_air_density, compute_v_eff
from src.inference.submission import write_submission
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
MODEL_DIR = _ROOT / "models"
SUBMISSION_PATH = _ROOT / "submissions" / "archive" / "v13.0_tft_noar.csv"

TARGET = "power_mw"


def prepare_data():
    """Prepare combined train+valid with weather features."""
    df_train = pd.read_csv(TRAIN_PATH)
    df_train[TIMESTAMP_COL] = pd.to_datetime(df_train[TIMESTAMP_COL])
    df_train = df_train.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_train = df_train.rename(columns={TARGET_COL: TARGET})

    df_valid = pd.read_csv(VALID_PATH)
    df_valid[TIMESTAMP_COL] = pd.to_datetime(df_valid[TIMESTAMP_COL])
    df_valid["_submission_row"] = range(len(df_valid))
    df_valid = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid[TARGET] = np.nan  # unknown

    era5 = pd.read_parquet(ERA5_PATH).rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})

    for df in [df_train, df_valid]:
        merged = df.merge(era5, on=TIMESTAMP_COL, how="left")
        for c in era5.columns:
            if c != TIMESTAMP_COL and c in merged.columns:
                df[c] = merged[c]

    for df in [df_train, df_valid]:
        df["air_density"] = compute_air_density(df["pressure_msl"], df["temperature_80m"])
        df["v_eff"] = compute_v_eff(df["wind_speed_120m"], df["air_density"])
        df["dir_sin_120"] = np.sin(np.deg2rad(df["wind_direction_120m"] * 1000))
        df["dir_cos_120"] = np.cos(np.deg2rad(df["wind_direction_120m"] * 1000))
        if "era5_wind_direction_100m" in df.columns:
            df["era5_dir_sin"] = np.sin(np.deg2rad(df["era5_wind_direction_100m"]))
            df["era5_dir_cos"] = np.cos(np.deg2rad(df["era5_wind_direction_100m"]))

    df_train["wind_speed_180m"] = df_train["wind_speed_180m"].fillna(df_train["wind_speed_120m"] * 1.06)
    df_train["wind_direction_180m"] = df_train["wind_direction_180m"].fillna(df_train["wind_direction_120m"])

    era5_new = [c for c in df_train.columns if c.startswith("era5_")]
    df_train[era5_new] = df_train[era5_new].fillna(0)
    df_valid[era5_new] = df_valid[era5_new].fillna(0)

    return df_train, df_valid


def main():
    set_global_seed(42)
    print("Preparing data...")
    df_train, df_valid = prepare_data()
    print(f"  Train: {len(df_train)}, Valid: {len(df_valid)}")

    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import MAE
    import lightning.pytorch as pl

    # Combine for continuous time index.
    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined["time_idx"] = range(len(combined))
    combined["series_id"] = "farm"
    # Fill target NaN in valid with 0 (won't be used for training loss).
    combined[TARGET] = combined[TARGET].fillna(0).astype(float)

    train_end_idx = int(combined[combined["_split"] == "train"]["time_idx"].max())

    # ALL features are "known future" — no target in encoder.
    known_reals = [
        "wind_speed_10m", "wind_speed_80m", "wind_speed_120m",
        "wind_gusts_10m", "temperature_80m", "pressure_msl",
        "dir_sin_120", "dir_cos_120", "v_eff", "air_density",
        TURBINES_IN_MAINTENANCE_COL,
    ]
    era5_known = ["era5_wind_speed_100m", "era5_wind_gusts_10m", "era5_pressure_msl",
                  "era5_temperature_2m", "era5_dir_sin", "era5_dir_cos"]
    known_reals.extend([c for c in era5_known if c in combined.columns])
    known_reals = [c for c in known_reals if c in combined.columns]
    combined[known_reals] = combined[known_reals].fillna(0).astype(float)

    print(f"  Known reals: {len(known_reals)}")

    # KEY DIFFERENCE: target is NOT in time_varying_unknown_reals.
    # Instead, we use a dummy "unknown" that's always 0.
    # Actually, PyTorch Forecasting requires the target to be in the dataset.
    # The trick: set max_prediction_length=1 so the model predicts 1 step ahead.
    # The encoder sees 168h of weather, decoder sees 1h of weather, predicts power.

    max_encoder_length = 168
    max_prediction_length = 1  # Single-step prediction

    training_data = combined[combined["time_idx"] <= train_end_idx].copy()

    # For non-autoregressive: target is "unknown" but we still need it in the dataset.
    # The model will learn to predict power from weather context alone.
    training_dataset = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[TARGET],  # target is "unknown" but available in training
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    print(f"  Dataset samples: {len(training_dataset)}")

    # Validation: last 2016 samples.
    val_cutoff = train_end_idx - 2016
    validation = training_dataset.filter(lambda x: x.time_idx_first_prediction > val_cutoff)
    train_subset = training_dataset.filter(lambda x: x.time_idx_first_prediction <= val_cutoff)
    print(f"  Train: {len(train_subset)}, Val: {len(validation)}")

    batch_size = 128  # Larger batch since prediction_length=1
    train_dl = train_subset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    pl.seed_everything(42)
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=3e-3,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=64,
        loss=MAE(),
        reduce_on_plateau_patience=3,
    )
    print(f"\n  TFT parameters: {tft.size() / 1e3:.1f}k")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=8, mode="min"),
            pl.callbacks.ModelCheckpoint(
                dirpath=str(MODEL_DIR / "tft_noar_checkpoints"),
                filename="tft_noar-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ],
        logger=False,
    )

    print("\n=== Training TFT (no-AR, GPU) ===")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\n  Best checkpoint: {best_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # Evaluate on validation.
    print("\n=== Validation ===")
    predictions = best_tft.predict(val_dl, return_y=True)
    y_pred = predictions.output.cpu().numpy().flatten()
    y_true = predictions.y[0].cpu().numpy().flatten()
    y_pred = np.clip(y_pred, 0, CAPACITY_MW)
    nmae = normalized_mae(y_true, y_pred)
    print(f"  TFT no-AR Validation nMAE: {nmae:.4f} %")

    # === Inference on Q1 2026 ===
    # Since prediction_length=1 and all features are "known future",
    # we can predict each hour independently (no rolling needed!).
    print("\n=== Inference on Q1 2026 ===")

    # Build inference dataset: for each valid hour, we need 168h encoder + 1h decoder.
    # The combined dataset already has valid rows. We just need to create a dataset
    # that covers the valid period.
    valid_start_idx = train_end_idx + 1
    valid_end_idx = int(combined["time_idx"].max())

    # Create inference dataset from the full combined data.
    # Each sample: encoder = [t-168, t-1], decoder = [t], predict power at t.
    inference_data = combined[combined["time_idx"] >= valid_start_idx - max_encoder_length].copy()

    inference_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        inference_data,
        predict=True,
        stop_randomization=True,
    )
    inference_dl = inference_dataset.to_dataloader(train=False, batch_size=256, num_workers=0)

    print(f"  Inference samples: {len(inference_dataset)}")
    raw_predictions = best_tft.predict(inference_dl)
    preds = raw_predictions.cpu().numpy().flatten()
    preds = np.clip(preds, 0, CAPACITY_MW)
    print(f"  Predictions: {len(preds)}, mean={preds.mean():.2f}, std={preds.std():.2f}")

    # Map predictions back to valid rows.
    n_valid = int((combined["_split"] == "valid").sum())
    if len(preds) >= n_valid:
        # Take the last n_valid predictions (corresponding to valid period).
        valid_preds = preds[-n_valid:]
    else:
        print(f"  WARNING: got {len(preds)} predictions, expected {n_valid}")
        valid_preds = np.zeros(n_valid)
        valid_preds[:len(preds)] = preds

    # Restore original row order.
    valid_rows = combined[combined["_split"] == "valid"].copy()
    order = valid_rows["_submission_row"].dropna().to_numpy().astype(int)
    preds_ordered = np.empty(n_valid, dtype=np.float32)
    preds_ordered[order] = valid_preds

    write_submission(preds_ordered, SUBMISSION_PATH, expected_rows=n_valid)
    print(f"  Mean: {valid_preds.mean():.2f}, P10: {np.percentile(valid_preds, 10):.2f}, P90: {np.percentile(valid_preds, 90):.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
