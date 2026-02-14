# trading_project/config/sentiment_config.py

"""
Конфandгурацandя for аналandwithу сентименту новин.
"""

SENTIMENT_DEFAULTS = {
    # Моwhereль for аналandwithу
    "model_name": "distilbert-base-uncased-finetuned-sst-2-english",  # DistilBERT як whereфолт
    # Пороги класифandкацandї
    "positive_threshold": 0.05,
    "negative_threshold": -0.05,
    "neutral_range": (-0.05, 0.05),
    # Ваги for рandwithних джерел
    "weights": {
        "rss": 1.0,
        "web": 1.0,
        "twitter": 0.8,
        "telegram": 0.8,
        "reddit": 0.9
    },
    # Опцandї нормалandforцandї
    "normalize": True,
    "scale": 1.0
}