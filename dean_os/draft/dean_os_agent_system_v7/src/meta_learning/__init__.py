# src/meta_learning/__init__.py

"""
Meta-Learning Package - Continuous Learning & Improvement System.
Пакет мета-навчання - Система безперервного навчання та вдосконалення.

Цей пакет відповідає за архітектурну пам'ять (Experience Diary),
розуміння ринкового контексту (Context Awareness) та еволюційні цикли
оптимізації стратегій (Learning Loops).
"""

# Lazy imports to avoid heavy dependencies (DuckDB, DataManager) on package import
# This allows import src.meta_learning without DuckDB installed

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


def __getattr__(name: str):
    """Lazy import for heavy meta-learning components."""
    if name == "EventImpact" or name == "EventType" or name == "MarketRegime":
        from .awareness.context.models import EventImpact, EventType, MarketRegime
        if name == "EventImpact":
            return EventImpact
        elif name == "EventType":
            return EventType
        elif name == "MarketRegime":
            return MarketRegime
    elif name == "ContextAwarenessEngine":
        from .awareness.context_engine import ContextAwarenessEngine
        return ContextAwarenessEngine
    elif name == "LearningLoopsEngine" or name == "TradingRule":
        from .evolution.dual_loops import LearningLoopsEngine, TradingRule
        if name == "LearningLoopsEngine":
            return LearningLoopsEngine
        elif name == "TradingRule":
            return TradingRule
    elif name == "DiaryEngine" or name == "DecisionRecord" or name == "DecisionType" or name == "DecisionOutcome":
        from .memory.diary_engine import DecisionOutcome, DecisionRecord, DecisionType, DiaryEngine
        if name == "DiaryEngine":
            return DiaryEngine
        elif name == "DecisionRecord":
            return DecisionRecord
        elif name == "DecisionType":
            return DecisionType
        elif name == "DecisionOutcome":
            return DecisionOutcome

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
