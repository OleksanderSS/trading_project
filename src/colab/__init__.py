"""Colab training module - refactored architecture with nested structure"""

# Import from subdirectories (nested structure)
from .memory import MemoryMonitor
from .utils import (
    get_optimal_batch_size,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    retry_on_timeout,
    compute_data_signature,
    compute_metrics
)
from .models import create_model
from .config import TrainingConfig, RuntimeConfigLoader, CheckpointParams
from .environment import ColabEnvironment

__all__ = [
    'MemoryMonitor',
    'get_optimal_batch_size',
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    'retry_on_timeout',
    'compute_data_signature',
    'compute_metrics',
    'create_model',
    'TrainingConfig',
    'RuntimeConfigLoader',
    'CheckpointParams',
    'ColabEnvironment'
]
