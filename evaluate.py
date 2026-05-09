"""Entry point for evaluating the trained LSTM forecaster on the test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.evaluation import collect_predictions, compute_metrics, plot_predictions
from src.model import LSTMForecaster
from src.utils import get_device, get_logger, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM forecaster on ETTh1 test set")
    parser.add_argument("--config", type=str, default="configs/horizon_96.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. If omitted, uses {paths.checkpoints_dir}/best.pt from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(cfg["paths"]["checkpoints_dir"]) / "best.pt"
    )

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("evaluate", log_file=results_dir / "evaluate.log")
    logger.info(f"Using device: {device}")

    _, _, test_loader, scaler = build_dataloaders(
        csv_path=cfg["data"]["csv_path"],
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    logger.info(f"Test set: {len(test_loader)} batches")

    model = LSTMForecaster(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        horizon=cfg["data"]["horizon"],
        dropout=cfg["model"]["dropout"],
        revin=cfg["model"].get("revin", False)
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.6f})"
    )

    y_pred, y_true = collect_predictions(model, test_loader, device)
    logger.info(f"Predictions shape: {y_pred.shape}")

    # Metrics in normalized space (comparable to training loss)
    norm_metrics = compute_metrics(y_pred, y_true)
    logger.info(
        f"Test (normalized): MSE={norm_metrics['mse']:.6f} | MAE={norm_metrics['mae']:.6f}"
    )

    # Metrics in original units (interpretable in °C)
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    y_true_denorm = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
    denorm_metrics = compute_metrics(y_pred_denorm, y_true_denorm)
    logger.info(
        f"Test (denormalized, °C): MSE={denorm_metrics['mse']:.4f} | "
        f"MAE={denorm_metrics['mae']:.4f}"
    )

    # Save metrics as JSON
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"normalized": norm_metrics, "denormalized": denorm_metrics}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()