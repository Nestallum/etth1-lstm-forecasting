"""Entry point for training the LSTM forecaster on ETTh1."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import build_dataloaders
from src.model import LSTMForecaster
from src.training import Trainer
from src.utils import get_device, get_logger, load_config, set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM forecaster on ETTh1")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    logs_dir = Path(cfg["paths"]["logs_dir"])
    logger = get_logger("train", log_file=logs_dir / "train.log")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _, _ = build_dataloaders(
        csv_path=cfg["data"]["csv_path"],
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    logger.info(
        f"Loaded {len(train_loader)} train batches, {len(val_loader)} val batches"
    )

    model = LSTMForecaster(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        horizon=cfg["data"]["horizon"],
        dropout=cfg["model"]["dropout"],
        revin=cfg["model"].get("revin", False)
    ).to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg["training"]["epochs"],
        grad_clip=cfg["training"]["grad_clip"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        checkpoints_dir=cfg["paths"]["checkpoints_dir"],
        logs_dir=cfg["paths"]["logs_dir"],
        logger=logger,
    )
    trainer.fit()
    logger.info("Training complete")


if __name__ == "__main__":
    main()