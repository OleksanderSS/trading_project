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

# Lazy imports to avoid heavy dependencies (advanced_engine, DEAN) on package import
# This allows import src.main.modes without triggering heavy module loads

__all__ = [
    'BaseMode',
    'TrainMode',
    'MonsterTestMode',
    'BacktestMode'
]


def __getattr__(name: str):
    """Lazy import for mode components."""
    if name == "BaseMode":
        from .base import BaseMode
        return BaseMode
    elif name == "BacktestMode":
        from .backtest import BacktestMode
        return BacktestMode
    elif name == "MonsterTestMode":
        from .monster_test import MonsterTestMode
        return MonsterTestMode
    elif name == "TrainMode":
        from .train import TrainMode
        return TrainMode

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
