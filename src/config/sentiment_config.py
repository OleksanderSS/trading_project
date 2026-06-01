# src/config/sentiment_config.py

"""
Configuration for news sentiment analysis.
"""

SENTIMENT_DEFAULTS = {
    # Model for analysis
    "model_name": "distilbert-base-uncased-finetuned-sst-2-english",  # DistilBERT as default
    # Classification thresholds
    "positive_threshold": 0.05,
    "negative_threshold": -0.05,
    "neutral_range": (-0.05, 0.05),
    # Weights for different sources
    "weights": {
        "rss": 1.0,
        "web": 1.0,
        "twitter": 0.8,
        "telegram": 0.8,
        "reddit": 0.9
    },
    # Normalization options
    "normalize": True,
    "scale": 1.0
}
