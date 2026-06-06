# src/feature_engineering/nlp/sentiment_core.py
import hashlib
import logging

from src.config.sentiment_config import SENTIMENT_DEFAULTS
from src.core.logging.logger import ProjectLogger
from src.features.nlp.scoring.news_score import compute_news_score


# ✅ Lazy import — transformers is heavy; loaded only when model is needed
def _get_auto_model_class():
    from transformers import AutoModelForSequenceClassification  # noqa: PLC0415
    return AutoModelForSequenceClassification

logger = ProjectLogger.get_logger("TradingProjectLogger")

def make_sentiment_key(text: str) -> str:
    # Use SHA-256 instead of MD5 for better security
    key = "sent_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[sentiment_score] Generated key for text: {key}")
    return key

def get_model():
    """
    Loads FinBERT as a pure model (AutoModelForSequenceClassification),
    so that it can be called with tokenized tensors.
    """
    model_name = SENTIMENT_DEFAULTS.get("model_name", "yiyanghkust/finbert-tone")
    logger.info(f"[sentiment_score] Loading model: {model_name}")
    return _get_auto_model_class().from_pretrained(model_name)

def compute_score(label: str, score: float) -> dict:
    label = label.lower()
    result = {
        "positive": score if label == "positive" else 0,
        "negative": score if label == "negative" else 0,
        "neutral": score if label == "neutral" else 0,
    }
    if label not in result:
        logger.warning(f"[sentiment_score] [WARN] Invalid label: {label}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[sentiment_score] Computed score: {result}")
    return result

def compute_news_score_safe(label: str, score: float, keywords: list) -> float:
    sentiment_dict = compute_score(label, score)
    if not keywords:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[sentiment_score] [DEBUG] Keywords are empty, score is computed without them")
        # Return a simple float representation: positive if positive, negative if negative
        if label.lower() == "positive":
            return float(score)
        elif label.lower() == "negative":
            return float(-score)
        return 0.0

    result = compute_news_score(sentiment_dict, keywords)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[sentiment_score] Final news_score: {result}")
    return float(result)
