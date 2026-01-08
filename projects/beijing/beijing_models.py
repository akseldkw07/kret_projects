import typing as t

import torch
import torch.nn as nn
from kret_lightning import *


class BeijingAirQualityLSTM(BaseLightningNN, CallbackMixin):
    """
    LSTM-based model for predicting PM2.5 air quality from temporal meteorological data.

    Architecture:
    - Embedding for categorical features (wind direction)
    - LSTM layers for temporal modeling
    - Fully connected layers for final prediction

    Features:
    - year, month, day, hour (temporal)
    - DEWP, TEMP, PRES (continuous meteorological)
    - cbwd (categorical - wind direction: NE, NW, SE, SW)
    - Iws, Is, Ir (cumulative wind speed, snow, rain)
    """

    version: str = "v000"

    # Wind direction embedding (cbwd is categorical)
    num_wind_directions: int = 4
    embedding_dim: int = 8

    def __init__(
        self,
        seq_length: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_cont_features: int = 14,
        **kwargs: t.Unpack[HPasKwargs],
    ):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_cont_features = num_cont_features

        # Wind direction embedding (cbwd is categorical)
        self.wind_embedding = nn.Embedding(self.num_wind_directions, self.embedding_dim)

        # Feature dimensions:
        # - 14 continuous features
        # - 8 from wind direction embedding
        lstm_input_size = num_cont_features + self.wind_embedding.embedding_dim
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected layers for prediction
        self.model = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),  # Single output: PM2.5 concentration
        )

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)
               Features expected in order:
               [year, month, day, hour, DEWP, TEMP, PRES, Iws, Is, Ir, cbwd_idx]
            debug: If True, print detailed information about each step

        Returns:
            Predictions of shape (batch_size, 1)
        """

        # Split continuous and categorical features
        # cbwd_idx is at the LAST position (index 10) after preprocessing
        continuous = x[:, :, :-1]  # All 10 continuous features
        cbwd_idx = x[:, :, -1].long()  # Wind direction is the last column

        # Embed wind direction
        # Fix for MPS device: ensure embedding weight is contiguous
        wind_embed = self.wind_embedding(cbwd_idx)  # (batch, seq_length, 8)
        if x.device.type == "mps":
            wind_embed = wind_embed.contiguous()
            continuous = continuous.contiguous()

        # Concatenate continuous features with wind embedding
        lstm_input = torch.cat([continuous, wind_embed], dim=2)

        # Defensive check: make sure preprocessing matches model config
        expected = self.lstm.input_size
        got = lstm_input.size(-1)
        assert got == expected, (
            f"Beijing LSTM input mismatch: got feature dim {got}, expected {expected}. "
            f"(continuous={continuous.size(-1)}, emb={wind_embed.size(-1)}). "
            f"Either set model_cfg.num_cont_features={continuous.size(-1)} or update preprocessing."
        )
        if x.device.type == "mps":
            lstm_input = lstm_input.contiguous()

        # LSTM forward pass
        lstm_out, _ = self.lstm(lstm_input)

        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Prediction
        output: torch.Tensor = self.model(last_hidden)  # (batch, 1)

        return output.squeeze(-1)


# ========================================================================================================================
# OLD STUFF
# ========================================================================================================================
