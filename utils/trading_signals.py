# utils/trading_signals.py

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from core.trading_advisor import TradingAdvisor
from utils.signal_utils import numeric_to_signal, SIGNAL_TO_NUM
from utils.logger import ProjectLogger
from typing import List, Optional
from typing import Optional


logger = ProjectLogger.get_logger("TradingProjectLogger")
# -------------------------------
# DEFAULT THRESHOLDS
# -------------------------------
DEFAULT_FORECAST_THRESHOLDS = {"bullish": 0.01, "bearish": -0.01}
DEFAULT_RSI_THRESHOLDS = {"oversold": 30, "overbought": 70}
DEFAULT_SENTIMENT_THRESHOLDS = {"positive": 0.2, "negative": -0.2}

# -------------------------------
# HELPERS
# -------------------------------
def normalize_sentiment(avg_sentiment: Optional[dict]) -> dict:
    if avg_sentiment is None:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    elif isinstance(avg_sentiment, (float, int)):
        return {"positive": float(avg_sentiment), "negative": 0.0, "neutral": 0.0}
    elif isinstance(avg_sentiment, dict):
        keys = ["positive", "negative", "neutral"]
        return {k: float(avg_sentiment.get(k, 0.0)) for k in keys}
    else:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

def signals_to_numeric(signals: Dict[str, str]) -> Dict[str, int]:
    return {k: SIGNAL_TO_NUM.get(str(v).upper(), 0) for k, v in signals.items()}

def fill_missing_data(df: pd.DataFrame, price_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if price_cols is None:
        price_cols = ["close", "open", "high", "low"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    return df

# -------------------------------
# MAIN SIGNALS
# -------------------------------
def generate_signals(
    df: Optional[pd.DataFrame],
    avg_sentiment: Optional[dict] = None,
    models_dict: Optional[dict] = None,
    forecast_thresholds: dict = DEFAULT_FORECAST_THRESHOLDS,
    rsi_thresholds: dict = DEFAULT_RSI_THRESHOLDS,
    sentiment_thresholds: dict = DEFAULT_SENTIMENT_THRESHOLDS,
    debug: bool = False,
    ticker: Optional[str] = None,  # додано параметр ticker
    thresholds_dict: Optional[dict] = None  # словник ticker -> thresholds
) -> Dict[str, str]:
    """Геnotрує сигнали BUY/HOLD/SELL череwith TradingAdvisor"""
    if df is None or df.empty:
        return {"forecast_signal": "HOLD",
                "rsi_signal": "HOLD",
                "sentiment_signal": "HOLD",
                "final_signal": "HOLD"}

    df = fill_missing_data(df)
    avg_sentiment = normalize_sentiment(avg_sentiment)

    # Якщо переданий ticker and thresholds_dict, пandдтягуємо thresholds по ньому
    if ticker and thresholds_dict:
        t_thresholds = thresholds_dict.get(ticker, {})
        forecast_thresholds = t_thresholds.get("forecast", forecast_thresholds)
        rsi_thresholds = t_thresholds.get("rsi", rsi_thresholds)
        sentiment_thresholds = t_thresholds.get("sentiment", sentiment_thresholds)

    advisor = TradingAdvisor(
        forecast_thresholds=forecast_thresholds,
        rsi_thresholds=rsi_thresholds,
        sentiment_thresholds=sentiment_thresholds
    )

    # Передаємо ticker обов'яwithково
    signals = advisor.get_signals_for_ticker(
        df_features=df,
        ticker=ticker,
        avg_sentiment=avg_sentiment
    )

    # Обробка NaN/None/notочandкуваних типandв
    for k, v in signals.items():
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).upper() not in SIGNAL_TO_NUM:
            signals[k] = "HOLD"

    if debug:
        logger.info(f"[DEBUG] Signals ({ticker}): {signals}")

    return signals

def aggregate_signals(signals_list: List[Dict[str, str]]) -> Dict[str, str]:
    if not signals_list:
        return {"forecast_signal": "HOLD",
                "rsi_signal": "HOLD",
                "sentiment_signal": "HOLD",
                "final_signal": "HOLD"}

    numeric_list = [signals_to_numeric(s) for s in signals_list]
    agg_signal = {}
    for key in signals_list[0].keys():
        values = [s.get(key, 0) for s in numeric_list]
        avg = sum(values) / len(values)
        agg_signal[key] = numeric_to_signal(avg)
    return agg_signal

def add_ensemble_signal(
    agg_signal: Dict[str, str],
    df: Optional[pd.DataFrame],
    models_dict: Optional[dict] = None,
    avg_sentiment: Optional[dict] = None,
    forecast_thresholds: dict = DEFAULT_FORECAST_THRESHOLDS,
    rsi_thresholds: dict = DEFAULT_RSI_THRESHOLDS,
    sentiment_thresholds: dict = DEFAULT_SENTIMENT_THRESHOLDS,
    debug: bool = False,
    ticker: Optional[str] = None  # <- додано
) -> Dict[str, str]:
    if df is None or df.empty:
        agg_signal["ensemble"] = "HOLD"
        return agg_signal

    df = fill_missing_data(df)
    avg_sentiment = normalize_sentiment(avg_sentiment)

    advisor = TradingAdvisor(
        models_dict=models_dict or {},
        forecast_thresholds=forecast_thresholds,
        rsi_thresholds=rsi_thresholds,
        sentiment_thresholds=sentiment_thresholds
    )

    # Передаємо ticker обов'яwithково
    ensemble_signal = advisor.get_signals_for_ticker(
        df_features=df,
        ticker=ticker,
        avg_sentiment=avg_sentiment
    ).get("final_signal", "HOLD")

    agg_signal["ensemble"] = ensemble_signal

    if debug:
        logger.info(f"[DEBUG] Ensemble signal: {ensemble_signal}")

    return agg_signal

def generate_signals_for_all_tickers(
    df_dict: Dict[str, Dict[str, pd.DataFrame]],
    avg_sentiment: Optional[dict] = None,
    models_dict: Optional[dict] = None,
    thresholds_dict: Optional[dict] = None,  # словник ticker -> thresholds
    debug: bool = False
) -> Dict[str, Dict[str, Dict[str, str]]]:
    all_signals = {}
    for ticker, tf_dict in df_dict.items():
        all_signals[ticker] = {}
        t_thresholds = thresholds_dict.get(ticker, {}) if thresholds_dict else {}
        for tf, df in tf_dict.items():
            signals = generate_signals(
                df,
                avg_sentiment=avg_sentiment,
                models_dict=models_dict,
                forecast_thresholds=t_thresholds.get("forecast", DEFAULT_FORECAST_THRESHOLDS),
                rsi_thresholds=t_thresholds.get("rsi", DEFAULT_RSI_THRESHOLDS),
                sentiment_thresholds=t_thresholds.get("sentiment", DEFAULT_SENTIMENT_THRESHOLDS),
                debug=debug,
                ticker=ticker,
                thresholds_dict=thresholds_dict
            )
            all_signals[ticker][tf] = signals
    return all_signals
