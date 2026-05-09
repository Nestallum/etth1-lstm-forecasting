"""Evaluation utilities: metrics and prediction plots for the LSTM forecaster."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a loader and return all predictions and targets.

    Returns:
        y_pred: array of shape (N, horizon)
        y_true: array of shape (N, horizon)
    """
    model.eval()
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        y_pred = model(x).cpu().numpy()
        preds.append(y_pred)
        targets.append(y.numpy())

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Compute MSE and MAE between predictions and targets."""
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    return {"mse": float(mse), "mae": float(mae)}


def plot_single_sample_with_history(
    window: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ax: plt.Axes,
    title: str = "",
) -> None:
    """Plot one sample with input history, prediction, and ground truth.

    Args:
        window: Input window in original units, shape (lookback,).
        y_pred: Predicted values in original units, shape (horizon,).
        y_true: True values in original units, shape (horizon,).
        ax: Matplotlib axis to plot into.
        title: Optional title for the subplot.
    """
    lookback = len(window)
    horizon = len(y_pred)
    t_input = np.arange(-lookback, 0)
    t_forecast = np.arange(0, horizon)

    ax.plot(t_input, window, label="Input window", color="steelblue", linewidth=1.2)
    ax.plot(t_forecast, y_true, label="Ground truth", color="seagreen", linewidth=1.5)
    ax.plot(
        t_forecast, y_pred, label="Prediction", color="crimson",
        linewidth=1.5, linestyle="--",
    )
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("OT (°C)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)