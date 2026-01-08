from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kret_utils.constants_kret import KretConstants
from ucimlrepo import fetch_ucirepo

DATA_DIR = KretConstants.DATA_DIR / "beijing"

BEIJING_AIR_REPO_ID = 381
COLUMNS_BEIJING = ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]


@cache
def _load_beijing_air_quality_data(id: int = BEIJING_AIR_REPO_ID):
    beijing_pm2_5 = fetch_ucirepo(id=id)

    # The data (features and target) are now pandas DataFrames
    X = beijing_pm2_5.data.features  # type: ignore
    y = beijing_pm2_5.data.targets  # type: ignore

    assert isinstance(X, pd.DataFrame)  # Features (Meteorological data)
    assert isinstance(y, pd.DataFrame)  # Target (PM2.5 concentration)

    return X.copy(deep=True), y.copy(deep=True)


def load_beijing_air_quality_data_reload(id: int = BEIJING_AIR_REPO_ID):
    X, y = _load_beijing_air_quality_data(id=id)
    return X.copy(deep=True), y.copy(deep=True)


def load_beijing_air_quality_data(data_dir: Path = DATA_DIR):
    raw = pd.read_csv(data_dir / "beijing_air.csv")
    Y = raw[["pm2.5"]]
    X = raw.drop(columns=["No"])
    return X.copy(deep=True), Y.copy(deep=True)


def create_sequences(tensor_tuple: tuple[torch.Tensor, torch.Tensor], seq_length: int = 24):
    """
    Create sequences from temporal data with proper preprocessing.

    IMPORTANT: To avoid data leakage, split your data BEFORE calling this function:
    """

    X_processed, y_processed = tensor_tuple

    # Create sequences
    X_sequences = []
    y_sequences = []

    for i in range(len(X_processed) - seq_length):
        X_seq = X_processed[i : i + seq_length]
        y_seq = y_processed[i + seq_length]  # Predict next timestep

        # Skip if any NaN in sequence
        if not (np.isnan(X_seq).any() or np.isnan(y_seq).any()):
            X_sequences.append(X_seq.numpy())
            y_sequences.append(y_seq.numpy())

    X_seq = np.array(X_sequences)  # (num_samples, seq_length, num_features)
    y_seq = np.array(y_sequences).flatten()  # (num_samples)

    # Final NaN check (using regular print since this runs before training loop)
    print(f"Created {len(X_seq)} sequences")
    print(f"X shape: {X_seq.shape}, contains NaN: {np.isnan(X_seq).any()}")
    print(f"y shape: {y_seq.shape}, contains NaN: {np.isnan(y_seq).any()}")
    print(f"X range: [{X_seq.min():.2f}, {X_seq.max():.2f}]")
    print(f"y range: [{y_seq.min():.2f}, {y_seq.max():.2f}]")

    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)
