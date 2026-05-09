"""LSTM-based forecaster with reversible instance normalization (RevIN)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Stacked LSTM encoder followed by a linear projection head.

    Optionally applies reversible instance normalization (RevIN): each input
    window is z-scored using its own statistics before being fed to the LSTM,
    and the prediction is rescaled with those same statistics. This addresses
    distribution shift between training and inference periods, a common
    weakness of standard normalization on long-term forecasting benchmarks.

    Reference:
        Kim et al., "Reversible Instance Normalization for Accurate
        Time-Series Forecasting against Distribution Shift", ICLR 2022.

    Shape conventions:
        input  : (batch, lookback, input_size)
        output : (batch, horizon)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.0,
        revin: bool = False,
        revin_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.revin = revin
        self.revin_eps = revin_eps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, input_size).

        Returns:
            Tensor of shape (batch, horizon).
        """
        if self.revin:
            # Compute per-instance statistics on the target feature (index 0
            # in univariate). For multivariate, this would normalize each
            # feature independently.
            mean = x.mean(dim=1, keepdim=True)              # (batch, 1, F)
            std = x.std(dim=1, keepdim=True) + self.revin_eps
            x = (x - mean) / std

        output, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        y = self.head(last_hidden)                          # (batch, horizon)

        if self.revin:
            # Denormalize using the input's mean/std for the target feature.
            # Squeeze feature dim since target is univariate (index 0).
            y = y * std[:, 0, 0:1] + mean[:, 0, 0:1]

        return y