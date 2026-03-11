# src/meta_learning/__init__.py

"""
Meta-Learning Package - Continuous Learning & Improvement System.
Пакет мета-навчання - Система безперервного навчання та вдосконалення.

Цей пакет відповідає за архітектурну пам'ять (Experience Diary), 
розуміння ринкового контексту (Context Awareness) та еволюційні цикли 
оптимізації стратегій (Learning Loops).
"""

from .memory.diary_engine import (
    DiaryEngine,
    DecisionRecord,
    DecisionType,
    DecisionOutcome
)
from .awareness.context_engine import (
    ContextAwarenessEngine,
    MarketRegime,
    EventType,
    EventImpact
)
from .evolution.dual_loops import (
    LearningLoopsEngine,
    TradingRule
)

__all__ = [
    "DiaryEngine",
    "DecisionRecord",
    "DecisionType",
    "DecisionOutcome",
    "ContextAwarenessEngine",
    "MarketRegime",
    "EventType",
    "EventImpact",
    "LearningLoopsEngine",
    "TradingRule"
]

__version__ = "1.1.0"
__author__ = "Dean Agent Architecture"