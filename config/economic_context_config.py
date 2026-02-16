
"""
Configuration for the Economic Context Map.

This file defines the rules for identifying different market regimes. The system
will evaluate these rules against market data to provide a high-level context
(e.g., 'bull_market') as a feature for the models.
"""

# Definitions for market context rules.
# These can be simple comparisons or complex formulas involving multiple indicators.
# The logic will be parsed and evaluated dynamically.
CONTEXT_DEFINITIONS = {
    # --- Trend-Based Contexts ---
    "bull_market_strong": {
        "description": "Strong Bull Market: Price is well above long-term average and RSI confirms strength.",
        "rule": "sma(50) > sma(200) AND rsi(14) > 55 AND close > sma(50)"
    },
    "bear_market_strong": {
        "description": "Strong Bear Market: Price is well below long-term average and RSI confirms weakness.",
        "rule": "sma(50) < sma(200) AND rsi(14) < 45 AND close < sma(50)"
    },
    "sideways_market_range_bound": {
        "description": "Range-bound market with low directional momentum.",
        "rule": "abs(sma(20) - sma(50)) / sma(50) < 0.01 AND rsi(14) > 40 AND rsi(14) < 60"
    },

    # --- Volatility-Based Contexts ---
    "high_volatility_expansion": {
        "description": "Period of high and expanding volatility.",
        "rule": "atr(14) > atr(14).rolling(50).mean() * 1.5"
    },
    "low_volatility_contraction": {
        "description": "Period of unusually low volatility, often preceding a breakout.",
        "rule": "atr(14) < atr(14).rolling(50).mean() * 0.75"
    },

    # --- Short-Term Momentum Contexts (as per your suggestion) ---
    "short_term_momentum_up": {
        "description": "Simple positive momentum from the previous candle.",
        "rule": "close > close.shift(1)"
    },
    "short_term_momentum_down": {
        "description": "Simple negative momentum from the previous candle.",
        "rule": "close < close.shift(1)"
    }
}
