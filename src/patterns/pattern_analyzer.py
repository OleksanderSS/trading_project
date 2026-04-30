import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from src.analytics.interfaces import IAnalyzer
from src.config.unified_config_manager import get_current_config

# Attempt to import pandas_ta for candle pattern recognition
try:
    import pandas_ta as ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)

class PatternDebugger:
    """Debug system for PatternAnalyzer execution tracking."""
    
    def __init__(self, enable_debug: bool = True):
        self.enable_debug = enable_debug
        self.debug_log = []
        self.start_time = None
        self.step_times = {}
        
        # Debug output directory
        self.debug_dir = Path("logs/pattern_debug")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        if self.enable_debug:
            logger.info("PatternDebugger initialized")
    
    def start_analysis(self, analysis_id: str = None):
        """Start debugging session."""
        if not self.enable_debug:
            return
            
        self.start_time = time.time()
        self.debug_log = []
        self.step_times = {}
        
        session_id = analysis_id or f"pattern_{datetime.now().strftime('%H%M%S')}"
        self.log("DEBUG", f"Pattern analysis started: {session_id}")
        self.log("INFO", f"Start time: {datetime.now().isoformat()}")
    
    def log(self, level: str, message: str, data: Any = None):
        """Log debug message with optional data."""
        if not self.enable_debug:
            return
            
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data
        }
        self.debug_log.append(log_entry)
        
        # Also log to standard logger
        log_msg = f"[PATTERN_DEBUG] {message}"
        if level == "DEBUG":
            logger.debug(log_msg)
        elif level == "INFO":
            logger.info(log_msg)
        elif level == "WARNING":
            logger.warning(log_msg)
        elif level == "ERROR":
            logger.error(log_msg)
    
    def track_step(self, step_name: str):
        """Track execution time for a step."""
        if not self.enable_debug:
            return
            
        current_time = time.time()
        if self.start_time:
            elapsed = current_time - self.start_time
            self.step_times[step_name] = elapsed
            self.log("DEBUG", f"Step '{step_name}' completed in {elapsed:.3f}s")
    
    def log_data_info(self, data_name: str, data: Any):
        """Log data shape and basic info."""
        if not self.enable_debug:
            return
            
        if isinstance(data, pd.DataFrame):
            info = {
                "type": "DataFrame",
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "null_count": data.isnull().sum().sum(),
                "sample": data.head(1).to_dict() if not data.empty else None
            }
        elif isinstance(data, list):
            info = {
                "type": "List",
                "length": len(data),
                "sample": data[0] if data else None
            }
        elif isinstance(data, dict):
            info = {
                "type": "Dict",
                "keys": list(data.keys()),
                "sample": list(data.items())[:3] if data else None
            }
        else:
            info = {
                "type": str(type(data).__name__),
                "value": str(data)
            }
        
        self.log("DEBUG", f"Data info for '{data_name}'", info)
    
    def log_patterns(self, patterns: Dict[str, Any], pattern_type: str):
        """Log detected patterns."""
        if not self.enable_debug:
            return
            
        if patterns:
            self.log("INFO", f"Detected {pattern_type} patterns: {list(patterns.keys())}")
            for pattern_name, pattern_value in patterns.items():
                self.log("DEBUG", f"{pattern_type} pattern '{pattern_name}': {pattern_value}")
        else:
            self.log("INFO", f"No {pattern_type} patterns detected")
    
    def log_bias_calculation(self, price_patterns: Dict, news_patterns: Dict, final_bias: float):
        """Log bias calculation details."""
        if not self.enable_debug:
            return
            
        bias_info = {
            "price_patterns": price_patterns,
            "news_patterns": news_patterns,
            "final_bias": final_bias,
            "calculation_steps": []
        }
        
        # Show calculation steps
        bias = 0.0
        for p, val in price_patterns.items():
            contribution = val / 500.0
            bias += contribution
            bias_info["calculation_steps"].append({
                "step": f"Price pattern '{p}'",
                "value": val,
                "contribution": contribution,
                "running_total": bias
            })
        
        if 'ai_euphoria' in news_patterns and bias > 0:
            old_bias = bias
            bias *= 1.2
            bias_info["calculation_steps"].append({
                "step": "AI euphoria synergy",
                "multiplier": 1.2,
                "before": old_bias,
                "after": bias
            })
        
        if 'geopolitical_risk' in news_patterns:
            old_bias = bias
            bias -= 0.3
            bias_info["calculation_steps"].append({
                "step": "Geopolitical risk adjustment",
                "adjustment": -0.3,
                "before": old_bias,
                "after": bias
            })
        
        final_bias = max(-1.0, min(1.0, bias))
        bias_info["final_bias"] = final_bias
        
        self.log("INFO", f"Bias calculation completed: {final_bias:.3f}", bias_info)
    
    def save_debug_session(self, results: Dict[str, Any] = None):
        """Save debug session to file."""
        if not self.enable_debug:
            return
            
        session_data = {
            "session_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "total_time": time.time() - self.start_time if self.start_time else None,
                "step_times": self.step_times
            },
            "debug_log": self.debug_log,
            "final_results": results
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = self.debug_dir / f"pattern_debug_{timestamp}.json"
        
        with open(debug_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.log("INFO", f"Debug session saved to: {debug_file}")
    
    def end_analysis(self, results: Dict[str, Any] = None):
        """End debugging session and save results."""
        if not self.enable_debug:
            return
            
        total_time = time.time() - self.start_time if self.start_time else 0
        self.log("INFO", f"Pattern analysis completed in {total_time:.3f}s")
        self.log("INFO", f"Total steps executed: {len(self.step_times)}")
        
        if results:
            self.log("INFO", "Final analysis results:", results)
        
        self.save_debug_session(results)

class PatternAnalyzer(IAnalyzer):
    """
    Advanced Pattern Analyzer that combines News Sentiment Patterns with Price Action.
    Implements IAnalyzer for integration with UnifiedAnalyticsEngine.
    """
    
    def __init__(self, enable_debug: bool = True):
        self.config = get_current_config().get('patterns', {})
        self.active_patterns = {}
        self.debugger = PatternDebugger(enable_debug=enable_debug)
        logger.info("PatternAnalyzer initialized with Price & News awareness.")

    def analyze(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main analysis method for UnifiedAnalyticsEngine.
        
        Args:
            data: Dictionary containing 'news_data' (List[Dict]) and 'price_data' (pd.DataFrame).
        """
        # Start debugging session
        self.debugger.start_analysis("pattern_analysis")
        
        news_list = data.get('news_data', [])
        price_df = data.get('price_data')
        
        self.debugger.log_data_info("input_data", data)
        self.debugger.log_data_info("news_list", news_list)
        self.debugger.log_data_info("price_df", price_df)
        
        results = {}
        
        # 1. Price Pattern Detection
        self.debugger.log("INFO", "Starting price pattern detection")
        price_patterns = {}
        if price_df is not None and not price_df.empty:
            price_patterns = self._detect_price_patterns(price_df)
            results['price_patterns'] = price_patterns
            self.debugger.log_patterns(price_patterns, "price")
        else:
            self.debugger.log("WARNING", "No price data available for pattern detection")
        self.debugger.track_step("price_pattern_detection")
            
        # 2. News Pattern Detection
        self.debugger.log("INFO", "Starting news pattern detection")
        news_patterns = {}
        if news_list:
            news_patterns = self._analyze_news_batch(news_list)
            results['news_patterns'] = news_patterns
            self.debugger.log_patterns(news_patterns, "news")
        else:
            self.debugger.log("WARNING", "No news data available for pattern detection")
        self.debugger.track_step("news_pattern_detection")
            
        # 3. Regime Detection
        self.debugger.log("INFO", "Starting regime pattern detection")
        market_metrics = data.get('market_metrics', {})
        self.debugger.log_data_info("market_metrics", market_metrics)
        regime_warnings = self.detect_regime_patterns(market_metrics, news_patterns)
        results['regime_warnings'] = regime_warnings
        results['market_state'] = "RISK_OFF" if regime_warnings else "RISK_ON"
        self.debugger.log("INFO", f"Market state determined: {results['market_state']}")
        self.debugger.log("DEBUG", f"Regime warnings: {regime_warnings}")
        self.debugger.track_step("regime_detection")
        
        # 4. Signal Bias Calculation
        self.debugger.log("INFO", "Starting signal bias calculation")
        signal_bias = self.get_signal_bias(price_patterns, news_patterns)
        results['signal_bias'] = signal_bias
        self.debugger.log_bias_calculation(price_patterns, news_patterns, signal_bias)
        self.debugger.track_step("signal_bias_calculation")
        
        # 5. Fractal Similarity (Optional)
        if price_df is not None and len(price_df) > 50:
            self.debugger.log("INFO", "Starting fractal similarity analysis")
            fractal_match = self._find_fractal_similarity(price_df)
            results['fractal_match'] = fractal_match
            self.debugger.log("DEBUG", f"Fractal similarity results: {fractal_match}")
        else:
            self.debugger.log("INFO", "Skipping fractal similarity - insufficient data")
        self.debugger.track_step("fractal_similarity_analysis")
            
        results['analysis_timestamp'] = datetime.now().isoformat()
        
        # End debugging session and save results
        self.debugger.end_analysis(results)
        
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
        # Enhanced theme detection with keyword-based analysis
        # Note: Future integration with NLPFeaturesEnricher or NewsAnalyzer for FinBERT/RoBERTa sentiment analysis
        themes = {'ai_euphoria': 0, 'rate_hike_stress': 0, 'geopolitical_risk': 0}
        for news in news_list:
            text = (news.get('text', '') + ' ' + news.get('title', '')).lower()
            # AI/Technology themes
            if 'ai' in text or 'nvidia' in text or 'artificial intelligence' in text:
                themes['ai_euphoria'] += 1
            # Federal Reserve/Interest Rate themes
            if 'fed' in text or 'rate hike' in text or 'federal reserve' in text:
                themes['rate_hike_stress'] += 1
            # Geopolitical conflict themes
            if 'conflict' in text or 'war' in text or 'geopolitical' in text:
                themes['geopolitical_risk'] += 1
            # Market volatility themes
            if 'volatility' in text or 'crash' in text or 'correction' in text:
                themes['market_volatility'] = themes.get('market_volatility', 0) + 1
            
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