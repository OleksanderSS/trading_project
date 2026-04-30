"""
Конфігурація для тренування моделей
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Конфігурація параметрів тренування"""
    model_type: str
    ticker: str
    target_col: str
    start_epoch: int
    epochs: int
    base_batch_size: int
    patience: int
    best_loss: float


@dataclass
class CheckpointParams:
    """Параметри для збереження контрольних точок"""
    model_path: str
    scaler_path: str
    optimizer_state: Optional[dict] = None
    epoch: int = 0


@dataclass
class TrainingParams:
    """Параметри для тренування моделі"""
    ticker: str
    target_col: str
    model_type: str
    input_size: int
    available_features: list


@dataclass
class TrainingLoopParams:
    """Параметри для циклу тренування"""
    model: object
    criterion: object
    optimizer: object
    data_dict: dict
    config: TrainingConfig
    checkpoint_params: CheckpointParams
