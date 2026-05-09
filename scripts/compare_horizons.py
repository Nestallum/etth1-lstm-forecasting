"""Compare LSTM forecasts across the four standard ETTh1 horizons.

Loads the four trained models (one per horizon) and plots their predictions
on a common starting point in the test set, with the same input history.
The result is a 4-panel figure showing how forecast quality degrades with
horizon length.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import HOURS_PER_MONTH, StandardScaler, load_etth1, split_temporal
from src.evaluation import plot_single_sample_with_history
from src.model import LSTMForecaster
from src.utils import get_device, load_config

@torch.no_grad()
def predict_window(
    model: LSTMForecaster,
    window: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    """Forecast the next `horizon` steps from a single input window."""
    window_norm = scaler.transform(window).astype(np.float32)
    x = torch.from_numpy(window_norm).unsqueeze(0).to(device)
    model.eval()
    y_pred_norm = model(x).cpu().numpy().squeeze(0)
    return scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

HORIZONS = [96, 192, 336, 720]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare forecasts across horizons")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the sample in the test set (relative to the start of test).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/images/horizons_comparison.png",
    )
    return parser.parse_args()


def load_model_for_horizon(horizon: int, device: torch.device) -> tuple[LSTMForecaster, dict]:
    """Load the trained model for a given horizon along with its config."""
    cfg_path = f"configs/horizon_{horizon}.yaml"
    cfg = load_config(cfg_path)

    model = LSTMForecaster(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        horizon=cfg["data"]["horizon"],
        dropout=cfg["model"]["dropout"],
        revin=cfg["model"].get("revin", False),
    ).to(device)

    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, cfg


def main() -> None:
    args = parse_args()
    device = get_device("auto")

    # Load shared data once
    cfg = load_config("configs/horizon_96.yaml")
    df = load_etth1(cfg["data"]["csv_path"])
    train_raw, _, _ = split_temporal(df)
    scaler = StandardScaler.fit(train_raw)

    lookback = cfg["data"]["lookback"]
    test_start = (12 + 4) * HOURS_PER_MONTH
    test_series = (
        df["OT"].to_numpy()[test_start:].reshape(-1, 1).astype(np.float32)
    )

    # Reference window (same starting point for all horizons)
    idx = args.sample_index
    if idx + lookback + max(HORIZONS) > len(test_series):
        raise ValueError(
            f"Sample index {idx} too large: history + max horizon "
            f"({lookback + max(HORIZONS)}) exceeds test set length ({len(test_series)})"
        )

    window = test_series[idx : idx + lookback]
    window_flat = window.flatten()

    # Build the figure
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(13, 11), sharex=False)

    for ax, horizon in zip(axes, HORIZONS):
        model, _ = load_model_for_horizon(horizon, device)
        forecast = predict_window(model, window, scaler, device)

        # Ground truth in °C (no scaling needed: test_series is in original units)
        y_true = test_series[idx + lookback : idx + lookback + horizon, 0]

        plot_single_sample_with_history(
            window=window_flat,
            y_pred=forecast,
            y_true=y_true,
            ax=ax,
            title=f"Horizon {horizon}h",
        )

    axes[-1].set_xlabel("Time step (hour, 0 = forecast start)")
    fig.suptitle(
        f"LSTM + RevIN forecasts across horizons (test sample #{idx})",
        fontsize=13,
    )
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()