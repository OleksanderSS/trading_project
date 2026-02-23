# utils/sentiment/thresholds.py

from typing import Dict
import pandas as pd
from utils.logger import ProjectLogger


logger = ProjectLogger.get_logger("TradingProjectLogger")

# -------------------- Сandтичнand пороги --------------------
FORECAST_THRESHOLDS = {
    "bullish": 0.01,  # мandнandмальний прогноwith for BUY
    "bearish": -0.01  # мandнandмальний прогноwith for SELL
}

RSI_THRESHOLDS = {
    "oversold": 30,   # нижня межа RSI for BUY
    "overbought": 70  # верхня межа RSI for SELL
}

SENTIMENT_THRESHOLDS = {
    "positive": 0.2,
    "negative": -0.2
}

# -------------------- Адаптивнand пороги --------------------
def compute_adaptive_sentiment_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Обчислює адаптивнand пороги сентименту for новинним DataFrame.
    Падає RuntimeError, якщо данand notкоректнand.
    """
    if df.empty:
        raise RuntimeError("[ERROR] compute_adaptive_sentiment_thresholds отримав пустий DataFrame")
    if "news_score" not in df.columns:
        raise RuntimeError("[ERROR] df not мandстить колонки 'news_score' for адаптивних порогandв")

    mean_score = df["news_score"].mean()
    std_score = df["news_score"].std()
    if pd.isna(std_score) or std_score == 0:
        raise RuntimeError("[ERROR] std_score notкоректний (0 or NaN) for адаптивних порогandв")

    return {
        "positive": mean_score + std_score,
        "negative": mean_score - std_score
    }
