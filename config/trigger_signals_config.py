# config/trigger_signals_config.py

SIGNAL_RULES = {
    "mention_spikes": {"action": "BUY", "weight": 1.0},
    "sentiment_extremes": {"action": "BUY", "weight": 1.2},
    "volatility_anomalies": {"action": "SELL", "weight": 1.0},
    "repeated_mentions": {"action": "HOLD", "weight": 0.5},
}