from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

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
    id: Optional[int]
    timestamp: datetime
    event_type: EventType
    title: str
    description: str
    source: str
    impact_level: EventImpact
    affected_tickers: List[str]
    affected_sectors: List[str]
    keywords: List[str]
    sentiment_score: float
    confidence: float
    relevance_score: float
    expiration_time: Optional[datetime]
    processed: bool
    impact_assessment: Dict[str, Any]

@dataclass
class MarketContext:
    """Market context snapshot."""
    timestamp: datetime
    market_regime: MarketRegime
    volatility_regime: str
    sentiment_index: float
    fear_greed_index: Optional[float]
    vix_level: Optional[float]
    major_events: List[MarketEvent]
    sector_performance: Dict[str, float]
    macro_indicators: Dict[str, float]
    risk_factors: List[str]
    opportunities: List[str]
    pattern_memory_insight: Optional[str] = None
