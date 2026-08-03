"""Training configuration classes"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TrainingConfig:
    """Configuration for training parameters"""
    def __init__(self, model_type, ticker, target_col, start_epoch, epochs,
                 base_batch_size, patience, best_loss):
        self.model_type = model_type
        self.ticker = ticker
        self.target_col = target_col
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.patience = patience
        self.best_loss = best_loss


@dataclass
class CheckpointParams:
    """Parameters for checkpoint saving"""
    ticker: str
    target_col: str
    m_type: str
    model: Any
    optimizer: Any
    epoch: int
    loss: float
    checkpoint_dir: Path


@dataclass
class TrainingParams:
    """Parameters for model training"""
    ticker: str
    target_col: str
    model_type: str
    features_df: Any
    y_ser: Any
    ticker_json: dict
    timeframe: str


@dataclass
class TrainingLoopParams:
    """Parameters for training loop execution"""
    model: Any
    criterion: Any
    optimizer: Any
    data_dict: dict
    model_type: str
    ticker: str
    target_col: str
