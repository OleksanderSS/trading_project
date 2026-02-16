# utils/sentiment_score.py

import hashlib
import pandas as pd
# from transformers import AutoModelForSequenceClassification # Modified by Gemini
from utils.sentiment.news_score import compute_news_score
from utils.logger import ProjectLogger
from config.sentiment_config import SENTIMENT_DEFAULTS

logger = ProjectLogger.get_logger("TradingProjectLogger")

def make_sentiment_key(text: str) -> str:
    key = "sent_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    logger.debug(f"[sentiment_score] Generated key for text: {key}")
    return key

def get_model():
    """
    Loads FinBERT as a pure model (AutoModelForSequenceClassification),
    so that it can be called with tokenized tensors.
    """
    # Modified by Gemini to fail gracefully
    raise ImportError("Transformers not available")
    # model_name = SENTIMENT_DEFAULTS.get("model_name", "yiyanghkust/finbert-tone")
    # logger.info(f"[sentiment_score] Loading model: {model_name}")
    # return AutoModelForSequenceClassification.from_pretrained(model_name)

def compute_score(label: str, score: float) -> dict:
    label = label.lower()
    result = {
        "positive": score if label == "positive" else 0,
        "negative": score if label == "negative" else 0,
        "neutral": score if label == "neutral" else 0,
    }
    if label not in result:
        logger.warning(f"[sentiment_score] [WARN] Invalid label: {label}")
    logger.debug(f"[sentiment_score] Computed score: {result}")
    return result

def compute_news_score_safe(label: str, score: float, keywords: list) -> float:
    if not keywords:
        logger.debug("[sentiment_score] [DEBUG] Keywords are empty, score is computed without them")
        # Return the basic score without keywords
        return compute_score(label, score)
    result = compute_news_score(compute_score(label, score), keywords)
    logger.debug(f"[sentiment_score] Final news_score: {result}")
    return result
