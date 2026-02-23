# utils/trigger_features.py

import pandas as pd
from triggers.trigger_layer import TriggerLayer
from utils.logger import ProjectLogger
from config.trigger_config import TRIGGER_DEFAULTS

logger = ProjectLogger.get_logger("TradingProjectLogger")

def compute_trigger_features(
    df_news: pd.DataFrame,
    df_prices: pd.DataFrame,
    mention_threshold: int = TRIGGER_DEFAULTS["mention_threshold"],
    sentiment_threshold: float = TRIGGER_DEFAULTS["sentiment_threshold"],
) -> dict:
    """
    Обчислює тригернand оwithнаки на основand новин.
    Поверandє словник Series:
    - mention_spikes
    - sentiment_extremes
    (volatility_anomalies прибрано як дублюючу цandль)
    """
    if df_news is None or df_news.empty:
        logger.warning("[trigger_features] [WARN] df_news порожнandй, поверandю пустand оwithнаки")
        return {
            "mention_spikes": pd.Series(dtype=int),
            "sentiment_extremes": pd.Series(dtype=float),
        }

    trigger_layer = TriggerLayer(df_news)

    trigger_data = {
        "mention_spikes": trigger_layer.detect_spike_mentions(threshold=mention_threshold),
        "sentiment_extremes": trigger_layer.detect_sentiment_extremes(threshold=sentiment_threshold),
    }

    logger.info(
        f"[trigger_features] [OK] Обчислено тригернand оwithнаки: "
        f"mentions={len(trigger_data['mention_spikes'])}, "
        f"sentiments={len(trigger_data['sentiment_extremes'])}"
    )

    return trigger_data