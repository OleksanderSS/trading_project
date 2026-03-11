import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from src.analytics.interfaces import IAnalyzer
from src.config.unified_config_manager import get_current_config

# Attempt to import pandas_ta for candle pattern recognition
try:
    import pandas_ta as ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)

class PatternAnalyzer(IAnalyzer):
    """
    Advanced Pattern Analyzer that combines News Sentiment Patterns with Price Action.
    Implements IAnalyzer for integration with UnifiedAnalyticsEngine.
    """
    
    def __init__(self):
        self.config = get_current_config().get('patterns', {})
        self.active_patterns = {}
        logger.info("PatternAnalyzer initialized with Price & News awareness.")

    def analyze(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main analysis method for UnifiedAnalyticsEngine.
        
        Args:
            data: Dictionary containing 'news_data' (List[Dict]) and 'price_data' (pd.DataFrame).
        """
        news_list = data.get('news_data', [])
        price_df = data.get('price_data')
        
        results = {}
        
        # 1. Price Pattern Detection
        price_patterns = {}
        if price_df is not None and not price_df.empty:
            price_patterns = self._detect_price_patterns(price_df)
            results['price_patterns'] = price_patterns
            
        # 2. News Pattern Detection (Simulated logic from previous version)
        news_patterns = {}
        if news_list:
            news_patterns = self._analyze_news_batch(news_list)
            results['news_patterns'] = news_patterns
            
        # 3. Regime Detection
        market_metrics = data.get('market_metrics', {})
        regime_warnings = self.detect_regime_patterns(market_metrics, news_patterns)
        results['regime_warnings'] = regime_warnings
        results['market_state'] = "RISK_OFF" if regime_warnings else "RISK_ON"
        
        # 4. Signal Bias Calculation
        results['signal_bias'] = self.get_signal_bias(price_patterns, news_patterns)
        
        # 5. Fractal Similarity (Optional)
        if price_df is not None and len(price_df) > 50:
            results['fractal_match'] = self._find_fractal_similarity(price_df)
            
        results['analysis_timestamp'] = datetime.now().isoformat()
        return results

    def _detect_price_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detects key candlestick and chart patterns."""
        patterns = {}
        
        if ta is not None:
            # Candlestick patterns using pandas_ta
            try:
                # Basic ones like Engulfing, Hammer, Doji
                cdl_df = df.ta.cdl_pattern(name=["engulfing", "hammer", "doji", "hangingman"])
                latest = cdl_df.iloc[-1]
                for col in cdl_df.columns:
                    if latest[col] != 0:
                        patterns[col] = int(latest[col]) # 100 for Bullish, -100 for Bearish
            except Exception as e:
                logger.warning(f"Failed to detect candle patterns: {e}")

        # Custom logic for Pin-bars
        body = abs(df['close'] - df['open'])
        range_total = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        # Bullish Pin-bar
        if lower_wick.iloc[-1] > body.iloc[-1] * 2 and upper_wick.iloc[-1] < body.iloc[-1]:
            patterns['bullish_pinbar'] = 100
            
        # Bearish Pin-bar
        if upper_wick.iloc[-1] > body.iloc[-1] * 2 and lower_wick.iloc[-1] < body.iloc[-1]:
            patterns['bearish_pinbar'] = -100

        # Double Top/Bottom (Simple check)
        window = 20
        if len(df) > window * 2:
            last_peaks = df['high'].rolling(window=window).max()
            if abs(last_peaks.iloc[-1] - last_peaks.iloc[-window]) / last_peaks.iloc[-1] < 0.002:
                patterns['potential_double_top'] = -1
                
        return patterns

    def _analyze_news_batch(self, news_list: List[Dict]) -> Dict:
        """Analyzes a batch of news for thematic patterns."""
        # TODO: Integrate with 'NLPFeaturesEnricher' or 'NewsAnalyzer' for more accurate theme detection using FinBERT/RoBERTa.
        themes = {'ai_euphoria': 0, 'rate_hike_stress': 0, 'geopolitical_risk': 0}
        for news in news_list:
            text = (news.get('text', '') + ' ' + news.get('title', '')).lower()
            if 'ai' in text or 'nvidia' in text: themes['ai_euphoria'] += 1
            if 'fed' in text or 'rate hike' in text: themes['rate_hike_stress'] += 1
            if 'conflict' in text or 'war' in text: themes['geopolitical_risk'] += 1
            
        return {k: v for k, v in themes.items() if v > 0}

    def get_signal_bias(self, price_patterns: Dict, news_patterns: Dict) -> float:
        """
        Calculates alignment between technical and fundamental patterns.
        Returns value between -1.0 (Strongly Bearish) and 1.0 (Strongly Bullish).
        """
        bias = 0.0
        
        # Tech bias
        for p, val in price_patterns.items():
            bias += (val / 500.0) # Normalize 100 to 0.2
            
        # News bias adjustment
        if 'ai_euphoria' in news_patterns and bias > 0:
            bias *= 1.2 # Synergy
        if 'geopolitical_risk' in news_patterns:
            bias -= 0.3 # Global weight
            
        return max(-1.0, min(1.0, bias))

    def _find_fractal_similarity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Finds the most similar sequence in recent history using simple Euclidean distance."""
        seq_len = 10
        current_seq = df['close'].pct_change().tail(seq_len).values
        
        if len(current_seq) < seq_len: return {}
        
        best_dist = float('inf')
        best_match_idx = -1
        
        # Scan last 500 candles
        search_range = min(len(df) - seq_len - 1, 500)
        for i in range(len(df) - search_range, len(df) - seq_len - 1):
            past_seq = df['close'].pct_change().iloc[i:i+seq_len].values
            dist = np.linalg.norm(current_seq - past_seq)
            if dist < best_dist:
                best_dist = dist
                best_match_idx = i
                
        if best_match_idx != -1:
            next_return = df['close'].pct_change().iloc[best_match_idx + seq_len]
            return {
                'similarity_score': 1 / (1 + best_dist),
                'historical_outcome': float(next_return),
                'match_timestamp': str(df.index[best_match_idx])
            }
        return {}

    def detect_regime_patterns(self, market_data: Dict, news_patterns: Dict) -> List[str]:
        """Identifies significant shifts in market regime."""
        warnings = []
        if market_data.get('vix', 0) > 30 or news_patterns.get('geopolitical_risk', 0) > 2:
            warnings.append("HIGH_VOLATILITY_REGIME")
        if market_data.get('tech_concentration', 0) > 0.7 and 'ai_euphoria' in news_patterns:
            warnings.append("SECTOR_BUBBLE_RISK")
        return warnings