import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from .models import MarketEvent, MarketContext, MarketRegime, EventImpact, EventType
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ContextAnalyzer")

class ContextAnalyzer:
    """Analyzes market events and conditions to determine context and regime."""
    
    def __init__(self, diary_engine: Any = None):
        self.diary = diary_engine

    def detect_market_regime(self, events: List[MarketEvent]) -> MarketRegime:
        """Detect current market regime based on event sentiment and type."""
        if not events: return MarketRegime.SIDEWAYS
        
        avg_sentiment = np.mean([e.sentiment_score for e in events])
        if avg_sentiment > 0.4: return MarketRegime.BULL_MARKET
        if avg_sentiment < -0.4: return MarketRegime.BEAR_MARKET
        
        # Check for volatility
        high_impact_count = sum(1 for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL])
        if high_impact_count > 3: return MarketRegime.VOLATILE
        
        return MarketRegime.SIDEWAYS

    def detect_volatility_regime(self, events: List[MarketEvent]) -> str:
        """Simple volatility regime detection."""
        high_impact = [e for e in events if e.impact_level in [EventImpact.HIGH, EventImpact.CRITICAL]]
        if len(high_impact) > 5: return 'extreme'
        if len(high_impact) > 2: return 'elevated'
        return 'normal'

    def calculate_sentiment_index(self, events: List[MarketEvent]) -> float:
        """Aggregate sentiment from multiple events."""
        if not events: return 0.5 # Neutral
        return float(np.mean([e.sentiment_score for e in events]))

    def get_memory_insight(self, current_pattern_id: Optional[str]) -> str:
        """Pattern Memory Sync: retrieves historical performance insight."""
        if not current_pattern_id or not self.diary:
            return "New regime detected."
            
        reliability = self.diary.get_pattern_reliability(current_pattern_id)
        if reliability < 0.4:
            return f"Pattern {current_pattern_id} has LOW historical win rate ({reliability:.1%}). CAUTION."
        elif reliability > 0.6:
            return f"Pattern {current_pattern_id} is STABLE ({reliability:.1%}). Models perform well."
        else:
            return f"Pattern {current_pattern_id} encountered previously. Neutral performance."

    def generate_recommendations(self, context: MarketContext) -> List[str]:
        """Generate actionable recommendations based on context."""
        recs = []
        if context.market_regime == MarketRegime.BEAR_MARKET:
            recs.append("Defensive positioning recommended.")
        elif context.market_regime == MarketRegime.BULL_MARKET:
            recs.append("Aggressive strategies favored.")
        
        if context.volatility_regime == 'extreme':
            recs.append("Reduce position sizes across all models.")
            
        return recs
