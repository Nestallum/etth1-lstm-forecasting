"""Training loop for the LSTM forecaster.

Wraps the train/validation cycle, gradient clipping, early stopping,
checkpointing of the best model, and TensorBoard logging.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Encapsulates the full training loop for a forecasting model.

    Tracks the best validation loss across epochs, saves the corresponding
    checkpoint, and stops early if validation does not improve for a given
    number of epochs (patience).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int,
        grad_clip: float,
        early_stopping_patience: int,
        checkpoints_dir: Path | str,
        logs_dir: Path | str,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger

        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoints_dir / "best.pt"

        self.writer = SummaryWriter(log_dir=logs_dir)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def fit(self) -> None:
        """Run the full training loop."""
        self.logger.info(
            f"Starting training: {self.epochs} epochs max, "
            f"early stopping patience {self.early_stopping_patience}"
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate()
            elapsed = time.time() - t0

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"time={elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss)
                self.logger.info("  -> new best val_loss, checkpoint saved")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(best val_loss={self.best_val_loss:.6f})"
                    )
                    break

        self.writer.close()

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch and return the average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d}",
            leave=False,
        )
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self) -> float:
        """Run one validation pass and return the average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save the current model state as the best checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            self.best_checkpoint_path,
        )