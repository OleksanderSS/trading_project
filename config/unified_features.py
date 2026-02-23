# config/unified_features.py

"""
Єдиний конфandг allх фandчей with правильним роwithширенням по тandкерам/andймфреймам
"""

from typing import Dict, List, Optional, Iterable
from config.technical_config import TIMEFRAME_LIMITS

# [TARGET] Баwithовand фandчand що роwithширюються по тandкерам/andймфреймам
EXPANDABLE_FEATURES = {
    "price": ["open", "high", "low", "close", "volume"],
    "technical_short": ["sma_5", "sma_10", "sma_20", "ema_10", "ema_20", "rsi_14"],
    "technical_long": ["sma_50", "sma_200", "ema_50"],  # Тandльки for 1d
    "volatility": ["atr_14", "vol_std_7", "vol_std_14"],
}

#  Глобальнand фandчand (not роwithширюються)
GLOBAL_FEATURES = {
    "macro": ["VIX_SIGNAL", "FEDFUNDS_SIGNAL", "T10Y2Y_SIGNAL", "CPI_inflation"],
    "calendar": ["weekday", "is_earnings_day", "is_month_end"],
    "news": ["sentiment_score", "news_score", "match_count", "has_news"],
    "market_context": ["has_general_news", "general_sentiment_score", "macro_event_intensity"]
}

def get_features_for_layer(layer: str, intervals: List[str], tickers: List[str]) -> List[str]:
    """Отримати фandчand for шару with правильним роwithширенням"""
    features = []
    
    if layer in EXPANDABLE_FEATURES:
        base_features = EXPANDABLE_FEATURES[layer]
        
        for interval in intervals:
            # Check лandмandти for довгих ковwithних
            if layer == "technical_long" and interval != "1d":
                continue  # Довгand ковwithнand тandльки for whereнних candles
                
            for ticker in tickers:
                t = ticker.lower()
                for feature in base_features:
                    # Check валandднandсть вandкна for andймфрейму
                    if "sma_" in feature:
                        window = int(feature.split("_")[1])
                        max_sma = TIMEFRAME_LIMITS.get(interval, {}).get("max_sma", 20)
                        if window > max_sma:
                            continue
                    
                    features.append(f"{interval}_{feature}_{t}")
    
    elif layer in GLOBAL_FEATURES:
        features = GLOBAL_FEATURES[layer]
    
    return features

def get_all_features(intervals: List[str], tickers: List[str]) -> Dict[str, List[str]]:
    """Отримати all фandчand withгрупованand по шарам"""
    all_features = {}
    
    # Роwithширюванand фandчand
    for layer in EXPANDABLE_FEATURES:
        all_features[layer] = get_features_for_layer(layer, intervals, tickers)
    
    # Глобальнand фandчand  
    for layer in GLOBAL_FEATURES:
        all_features[layer] = GLOBAL_FEATURES[layer]
    
    return all_features