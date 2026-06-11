from .backtest import BacktestMode
from .base import BaseMode
from .monster_test import MonsterTestMode
from .train import TrainMode

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

__all__ = [
    'BaseMode',
    'TrainMode',
    'MonsterTestMode',
    'BacktestMode'
]
