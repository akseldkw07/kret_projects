import typing as t

import torch
import torch.nn as nn
from kret_lightning.abc_lightning import HPasKwargs
from kret_lightning.base_lightning_nn import BaseLightningNN
from kret_lightning.mixin_callbacks import CallbackMixin
from kret_lightning.mixin_metrics import MetricMixin
from kret_lightning.mixin_optuna import OptunaMixin


class HeartFailureNN(MetricMixin, BaseLightningNN, CallbackMixin, OptunaMixin):
    """
    Neural network model for predicting heart failure events.

    Architecture:
    - Fully connected layers with ReLU activations
    - Dropout for regularization

    Features:
    - age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
      high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
    """

    version: str = "v_004"
    ignore_hparams = ["input_size"]
    _criterion = nn.BCEWithLogitsLoss()

    def __init__(
        self,
        input_size: int = 21,
        hidden_sizes: list[int] = [64, 32],
        dropout_rate: float = 0.3,
        **kwargs: t.Unpack[HPasKwargs],
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=self.ignore_hparams)
        self.setup_metrics(task="binary")
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate

        layers = []
        in_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()  # Squeeze for binary classification

    def predict_step(self, batch, batch_idx=0) -> torch.Tensor:
        """Returns binary predictions (0/1) by thresholding sigmoid probabilities."""
        probs = super().predict_step(batch, batch_idx)  # sigmoid already applied by MetricMixin
        return (probs >= 0.5).long()
