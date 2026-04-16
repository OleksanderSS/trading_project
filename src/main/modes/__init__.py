"""
Modes package - різні режими роботи системи
"""

"""
Modes package - різні режими роботи системи

Legacy modes (removed/consolidated in previous refactoring):
- AnalyzeMode → Merged into BacktestMode + MonsterTestMode  
- BatchTrainingMode → Replaced by src.training.batch_trainer + UnifiedTrainingManager
- ProgressiveMode → Replaced by src.training.progressive_trainer + UnifiedTrainingManager
"""

from .base import BaseMode
from .train import TrainMode
from .monster_test import MonsterTestMode
from .backtest import BacktestMode

__all__ = [
    'BaseMode',
    'TrainMode',
    'MonsterTestMode',
    'BacktestMode'
]
