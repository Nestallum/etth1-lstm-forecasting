"""Data pipeline for ETTh1 time series forecasting.

Handles CSV loading, temporal train/val/test splits (Informer convention),
z-score normalization (fit on train only), and sliding-window dataset
construction for multi-step ahead forecasting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Informer convention: months counted as 30-day blocks (720 hours)
HOURS_PER_MONTH = 30 * 24
TRAIN_HOURS = 12 * HOURS_PER_MONTH  # 8640
VAL_HOURS = 4 * HOURS_PER_MONTH     # 2880
TEST_HOURS = 4 * HOURS_PER_MONTH    # 2880


@dataclass
class StandardScaler:
    """Z-score normalizer fitted on training data only.

    Stores per-feature mean and std as numpy arrays. Applies the same
    transform to validation and test sets to prevent data leakage.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        """Compute mean and std along the time axis (axis=0)."""
        return cls(mean=x.mean(axis=0), std=x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean
    

def load_etth1(csv_path: Path | str) -> pd.DataFrame:
    """Load the ETTh1 CSV with parsed datetime index."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def split_temporal(
    df: pd.DataFrame, target_col: str = "OT"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the series into train/val/test using the Informer convention.

    Args:
        df: Full ETTh1 DataFrame with a 'date' column and feature columns.
        target_col: Name of the column to forecast (default: 'OT').

    Returns:
        Three numpy arrays of shape (T, 1), respectively for train, val, test,
        where T is the number of hours in each split.
    """
    series = df[target_col].to_numpy().reshape(-1, 1).astype(np.float32)

    train_end = TRAIN_HOURS
    val_end = TRAIN_HOURS + VAL_HOURS
    test_end = TRAIN_HOURS + VAL_HOURS + TEST_HOURS

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:test_end]

    return train, val, test


class ETTh1Dataset(Dataset):
    """Sliding-window dataset for multi-step ahead forecasting.

    Given a continuous time series of shape (T, F), produces all possible
    (input_window, target_window) pairs with stride 1.

    Each sample is:
        input  : tensor of shape (lookback, F)
        target : tensor of shape (horizon,)  — target column only

    The number of samples is T - lookback - horizon + 1.
    """

    def __init__(
        self,
        series: np.ndarray,
        lookback: int,
        horizon: int,
        target_index: int = 0,
    ) -> None:
        if series.ndim != 2:
            raise ValueError(f"Expected 2D array (T, F), got shape {series.shape}")
        if len(series) < lookback + horizon:
            raise ValueError(
                f"Series too short: {len(series)} < lookback + horizon = {lookback + horizon}"
            )

        self.series = torch.from_numpy(series)  # (T, F), float32
        self.lookback = lookback
        self.horizon = horizon
        self.target_index = target_index

    def __len__(self) -> int:
        return len(self.series) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.series[idx : idx + self.lookback]
        y = self.series[
            idx + self.lookback : idx + self.lookback + self.horizon,
            self.target_index,
        ]
        return x, y
    
    
def build_dataloaders(
    csv_path: Path | str,
    lookback: int,
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """End-to-end pipeline: CSV → splits → scaling → datasets → dataloaders.

    Args:
        csv_path: Path to ETTh1.csv.
        lookback: Length of the input window (hours).
        horizon: Length of the forecast window (hours).
        batch_size: Mini-batch size for all loaders.
        num_workers: PyTorch DataLoader workers (0 on Windows is safest).

    Returns:
        train_loader, val_loader, test_loader, scaler
        The scaler is returned so predictions can be inverse-transformed
        for interpretable metrics.
    """
    df = load_etth1(csv_path)
    train_raw, val_raw, test_raw = split_temporal(df)

    # Fit scaler on train only — critical to prevent data leakage
    scaler = StandardScaler.fit(train_raw)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)

    train_ds = ETTh1Dataset(train, lookback, horizon)
    val_ds = ETTh1Dataset(val, lookback, horizon)
    test_ds = ETTh1Dataset(test, lookback, horizon)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, scaler