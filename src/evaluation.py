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


def plot_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    output_path: Path | str,
    n_samples: int = 4,
    seed: int = 42,
) -> None:
    """Plot a few random prediction-vs-truth windows side by side.

    Args:
        y_pred: array of shape (N, horizon), predictions in original units.
        y_true: array of shape (N, horizon), targets in original units.
        output_path: where to save the figure.
        n_samples: number of random examples to plot.
        seed: for reproducible sample selection.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y_pred), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2.5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(y_true[idx], label="Ground truth", color="steelblue", linewidth=1.5)
        ax.plot(y_pred[idx], label="Prediction", color="crimson", linewidth=1.5, linestyle="--")
        ax.set_ylabel("OT (°C)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_title(f"Sample #{idx}")

    axes[-1].set_xlabel("Forecast step (hour)")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)