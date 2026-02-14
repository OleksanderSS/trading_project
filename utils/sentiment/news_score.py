# utils/sentiment/news_score.py

from typing import List, Dict
from utils.logger import ProjectLogger


logger = ProjectLogger.get_logger("TradingProjectLogger")

def compute_news_score(
    sentiment: Dict[str, float],
    keywords: List[str],
    weight_keywords: float = 0.7,
    weight_sentiment: float = 0.3
) -> float:
    """
    Обчислює рейтинг новини на основand keywords and сентименту.
    sentiment: dict with ключами "positive" and "negative" (0..1)
    keywords: список withнайwhereних keywords
    """
    if not isinstance(sentiment, dict):
        sentiment = {}
    if not isinstance(keywords, list):
        keywords = []

    pos = float(sentiment.get("positive", 0.0))
    neg = float(sentiment.get("negative", 0.0))

    # нормалandforцandя
    pos = max(0.0, min(pos, 1.0))
    neg = max(0.0, min(neg, 1.0))

    sentiment_score = abs(pos - neg)
    keyword_score = 1.0 if keywords else 0.0

    total_weight = weight_keywords + weight_sentiment
    if total_weight == 0:
        total_weight = 1.0  # forхист вandд дandлення на нуль

    return (weight_keywords * keyword_score + weight_sentiment * sentiment_score) / total_weight
