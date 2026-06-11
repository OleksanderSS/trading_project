from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Event types"""
    ECONOMIC_RELEASE = 'economic_release'
    CORPORATE_NEWS = 'corporate_news'
    MARKET_EVENT = 'market_event'
    GEOPOLITICAL = 'geopolitical'
    REGULATORY = 'regulatory'
    WEATHER = 'weather'
    SOCIAL_SENTIMENT = 'social_sentiment'

class EventImpact(Enum):
    """Event impact level"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class MarketRegime(Enum):
    """Market regimes"""
    BULL_MARKET = 'bull_market'
    BEAR_MARKET = 'bear_market'
    SIDEWAYS = 'sideways'
    VOLATILE = 'volatile'
    CRISIS = 'crisis'

@dataclass
class MarketEvent:
    """Market event data structure."""
    id: int | None
    timestamp: datetime
    event_type: EventType
    title: str
    description: str
    source: str
    impact_level: EventImpact
    affected_tickers: list[str]
    affected_sectors: list[str]
    keywords: list[str]
    sentiment_score: float
    confidence: float
    relevance_score: float
    expiration_time: datetime | None
    processed: bool
    impact_assessment: dict[str, Any]

@dataclass
class MarketContext:
    """Market context snapshot."""
    timestamp: datetime
    market_regime: MarketRegime
    volatility_regime: str
    sentiment_index: float
    fear_greed_index: float | None
    vix_level: float | None
    major_events: list[MarketEvent]
    sector_performance: dict[str, float]
    macro_indicators: dict[str, float]
    risk_factors: list[str]
    opportunities: list[str]
    pattern_memory_insight: str | None = None
