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

# TrainMode and PredictMode were archived on 2026-08-12: their dispatcher
# (src/archive/main/system_orchestrator.py) had already been archived, no root
# script invoked them, and the live pipeline trains in Stage 4 and predicts in
# Stage 5 through run_hybrid_pipeline.py. The modes still here are the ones
# with a live entry point: run_shadow_battle.py, run_historical_replay.py,
# run_monster_test.py and scripts/verify_backtesting.py.
__all__ = [
    'BaseMode',
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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
