"""LSTM-based forecaster for multi-step ahead time series prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Stacked LSTM encoder followed by a linear projection head.

    The LSTM consumes the lookback window and produces a sequence of hidden
    states. We take the last hidden state — a fixed-size summary of the
    entire input — and project it directly to the full forecast horizon.

    This multi-output approach avoids the exposure bias of autoregressive
    decoding while remaining competitive on benchmarks like ETTh1.

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
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

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
            Tensor of shape (batch, horizon) with normalized predictions.
        """
        # output: (batch, lookback, hidden_size)
        # h_n:    (num_layers, batch, hidden_size)
        output, (h_n, _) = self.lstm(x)

        # Take the last layer's final hidden state: (batch, hidden_size)
        last_hidden = h_n[-1]

        return self.head(last_hidden)