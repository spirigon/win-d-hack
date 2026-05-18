"""TFT training for wind power forecasting.

Uses PyTorch Forecasting's TFT with:
- Encoder: 168h of past weather + power
- Decoder: 24h of future weather (known)
- All weather is "known future" since it's available for Q1 2026
- Target is power_mw

For Q1 2026 inference: roll forward in 24h blocks.

Usage:
    python -m src.training.train_tft
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
from src.utils.seeding import set_global_seed

TRAIN_PATH = _ROOT / "data" / "raw" / "train_dataset.csv"
VALID_PATH = _ROOT / "data" / "raw" / "valid_features.csv"
ERA5_PATH = _ROOT / "data" / "external" / "era5_reanalysis.parquet"
MODEL_DIR = _ROOT / "models"

TARGET = "power_mw"  # renamed to avoid '.' in column name


def prepare_data():
    """Load and prepare data for TFT."""
    df_train = pd.read_csv(TRAIN_PATH)
    df_train[TIMESTAMP_COL] = pd.to_datetime(df_train[TIMESTAMP_COL])
    df_train = df_train.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_train = df_train.rename(columns={TARGET_COL: TARGET})

    df_valid = pd.read_csv(VALID_PATH)
    df_valid[TIMESTAMP_COL] = pd.to_datetime(df_valid[TIMESTAMP_COL])
    df_valid = df_valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df_valid[TARGET] = 0.0  # placeholder

    # ERA5.
    era5 = pd.read_parquet(ERA5_PATH).rename(columns={"time": TIMESTAMP_COL})
    era5_cols = [c for c in era5.columns if c != TIMESTAMP_COL]
    era5 = era5.rename(columns={c: f"era5_{c}" for c in era5_cols})

    for df in [df_train, df_valid]:
        df_merged = df.merge(era5, on=TIMESTAMP_COL, how="left")
        for c in era5.columns:
            if c != TIMESTAMP_COL:
                df[c] = df_merged[c]

    # Physics features.
    for df in [df_train, df_valid]:
        df["air_density"] = compute_air_density(df["pressure_msl"], df["temperature_80m"])
        df["v_eff"] = compute_v_eff(df["wind_speed_120m"], df["air_density"])
        df["dir_sin_120"] = np.sin(np.deg2rad(df["wind_direction_120m"] * 1000))
        df["dir_cos_120"] = np.cos(np.deg2rad(df["wind_direction_120m"] * 1000))
        if "era5_wind_direction_100m" in df.columns:
            df["era5_dir_sin"] = np.sin(np.deg2rad(df["era5_wind_direction_100m"]))
            df["era5_dir_cos"] = np.cos(np.deg2rad(df["era5_wind_direction_100m"]))

    # Impute.
    df_train["wind_speed_180m"] = df_train["wind_speed_180m"].fillna(df_train["wind_speed_120m"] * 1.06)
    df_train["wind_direction_180m"] = df_train["wind_direction_180m"].fillna(df_train["wind_direction_120m"])

    # Fill NaN.
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

    # Combine into single continuous series.
    df_train["_split"] = "train"
    df_valid["_split"] = "valid"
    combined = pd.concat([df_train, df_valid], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    combined["time_idx"] = range(len(combined))
    combined["series_id"] = "farm"

    train_end_idx = int(combined[combined["_split"] == "train"]["time_idx"].max())

    # Known future reals (weather — available for all timestamps).
    known_reals = [
        "wind_speed_10m", "wind_speed_80m", "wind_speed_120m",
        "wind_gusts_10m", "temperature_80m", "pressure_msl",
        "dir_sin_120", "dir_cos_120", "v_eff", "air_density",
        TURBINES_IN_MAINTENANCE_COL,
    ]
    # Add ERA5.
    era5_known = ["era5_wind_speed_100m", "era5_wind_gusts_10m", "era5_pressure_msl",
                  "era5_temperature_2m", "era5_dir_sin", "era5_dir_cos"]
    known_reals.extend([c for c in era5_known if c in combined.columns])
    known_reals = [c for c in known_reals if c in combined.columns]

    # Ensure numeric and no NaN.
    combined[known_reals] = combined[known_reals].fillna(0).astype(float)
    combined[TARGET] = combined[TARGET].fillna(0).astype(float)

    print(f"  Known reals: {len(known_reals)}")
    print(f"  Train end idx: {train_end_idx}")

    max_encoder_length = 168
    max_prediction_length = 24

    # Build dataset on training portion only.
    training_data = combined[combined["time_idx"] <= train_end_idx].copy()

    training_dataset = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[TARGET],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    print(f"  Dataset samples: {len(training_dataset)}")

    # Split: last 2016 samples for validation (Q1 2025 surrogate).
    val_cutoff = train_end_idx - 2016
    validation = training_dataset.filter(lambda x: x.time_idx_first_prediction > val_cutoff)
    train_subset = training_dataset.filter(lambda x: x.time_idx_first_prediction <= val_cutoff)
    print(f"  Train subset: {len(train_subset)}, Validation: {len(validation)}")

    batch_size = 64
    train_dl = train_subset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # TFT model.
    pl.seed_everything(42)
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=MAE(),
        reduce_on_plateau_patience=4,
    )
    print(f"\n  TFT parameters: {tft.size() / 1e3:.1f}k")

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            pl.callbacks.ModelCheckpoint(
                dirpath=str(MODEL_DIR / "tft_checkpoints"),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ],
        logger=False,
    )

    print("\n=== Training TFT (CPU, ~30 epochs) ===")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\n  Best checkpoint: {best_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

    # Evaluate.
    print("\n=== Validation evaluation ===")
    predictions = best_tft.predict(val_dl, return_y=True)
    y_pred = predictions.output.cpu().numpy()
    y_true = predictions.y[0].cpu().numpy()

    # Flatten (prediction_length=24, so we get blocks).
    y_pred_flat = np.clip(y_pred.flatten(), 0, CAPACITY_MW)
    y_true_flat = y_true.flatten()

    # Only evaluate where y_true > 0 (valid target).
    mask = y_true_flat > 0
    if mask.sum() > 0:
        nmae = normalized_mae(y_true_flat[mask], y_pred_flat[mask])
        print(f"  TFT Validation nMAE: {nmae:.4f} %")
    else:
        print("  No valid targets in validation set.")

    print(f"\n  Predictions shape: {y_pred.shape}")
    print(f"  Mean pred: {y_pred_flat.mean():.2f}, std: {y_pred_flat.std():.2f}")
    print("\nDone. TFT training complete.")


if __name__ == "__main__":
    main()
