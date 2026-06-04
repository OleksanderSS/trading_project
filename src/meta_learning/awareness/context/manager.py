from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.meta_learning.base import BaseMetaComponent

from .analyzer import ContextAnalyzer
from .models import MarketContext
from .scanner import EventScanner
from .storage import ContextStorage

logger = ProjectLogger.get_logger("ContextAwarenessEngine")

class ContextAwarenessEngine(BaseMetaComponent):
    """
    Real-time Context Awareness Engine.
    Modular implementation delegating to storage, analyzer, and scanner.
    """

    def __init__(self, db_path: str = 'realtime_context_awareness.db', diary_engine=None):
        self.logger = logger
        self.storage = ContextStorage(db_path)
        self.analyzer = ContextAnalyzer(diary_engine)
        self.scanner = EventScanner()
        self.diary = diary_engine

    @property
    def name(self) -> str:
        return 'context_awareness'

    def update(self, data: Any = None) -> None:
        """Update context by scanning news and analyzing conditions."""
        self.logger.info('Triggering modular context awareness update...')
        try:
            # 1. Scan for new events
            new_events = self.scanner.scan_all_sources()
            for event in new_events:
                self.storage.save_market_event(event)

            # 2. Analyze context
            pattern_id = data.get('context_pattern_id') if isinstance(data, dict) else None
            self.analyze_market_context(pattern_id)

            self.logger.info('Context awareness update completed.')
        except Exception as e:
            self.logger.error(f'Failed to update context awareness: {e}', exc_info=True)

    def analyze_market_context(self, current_pattern_id: str | None = None) -> MarketContext:
        """Analyze current context and sync with pattern memory."""
        recent_events = self.storage.get_recent_events(hours=24)

        regime = self.analyzer.detect_market_regime(recent_events)
        vol_regime = self.analyzer.detect_volatility_regime(recent_events)
        sentiment = self.analyzer.calculate_sentiment_index(recent_events)
        memory_insight = self.analyzer.get_memory_insight(current_pattern_id)
        fear_greed_index = self._estimate_fear_greed_index(sentiment)
        vix_level = self._estimate_vix_level(vol_regime)

        context = MarketContext(
            timestamp=datetime.now(),
            market_regime=regime,
            volatility_regime=vol_regime,
            sentiment_index=sentiment,
            fear_greed_index=fear_greed_index,
            vix_level=vix_level,
            major_events=recent_events[:10],
            sector_performance={},
            macro_indicators={},
            risk_factors=[],
            opportunities=[],
            pattern_memory_insight=memory_insight
        )

        self.storage.save_market_context(context)
        return context

    def _estimate_fear_greed_index(self, sentiment: float) -> float:
        """Estimate a 0-100 fear/greed score from aggregate event sentiment."""
        if sentiment < 0:
            score = (sentiment + 1.0) * 50.0
        else:
            score = sentiment * 100.0
        return max(0.0, min(100.0, float(score)))

    def _estimate_vix_level(self, volatility_regime: str) -> float:
        """Map internal volatility regime to an approximate VIX-like level."""
        return {
            'normal': 15.0,
            'elevated': 25.0,
            'extreme': 35.0,
        }.get(volatility_regime, 20.0)

    def get_state(self) -> dict[str, Any]:
        """Return engine summary state."""
        return {
            'name': self.name,
            'db_path': self.storage.db_path,
            'has_diary': self.diary is not None
        }

    def close(self):
        self.storage.close()
