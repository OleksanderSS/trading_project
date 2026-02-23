# utils/unified_signals.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
from utils.logger import ProjectLogger
from config.thresholds import (
    get_rsi_threshold,
    get_forecast_threshold,
    get_sentiment_threshold,
    get_insider_threshold
)
from config.signal_priority import SIGNAL_PRIORITY

logger = ProjectLogger.get_logger("UnifiedSignals")

SIGNAL_TO_NUM = {"BUY": 1, "HOLD": 0, "SELL": -1}

def numeric_to_signal(score: float, ticker: Optional[str] = None) -> str:
    """Мапandнг числового score  BUY/HOLD/SELL with урахуванням порогandв andwith thresholds.py"""
    if ticker:
        pos_thr = get_sentiment_threshold(ticker, "positive")
        neg_thr = get_sentiment_threshold(ticker, "negative")
    else:
        pos_thr, neg_thr = 0.2, -0.2

    if score > pos_thr:
        return "BUY"
    elif score < neg_thr:
        return "SELL"
    return "HOLD"

def signals_to_numeric(signals: Dict[str, str]) -> Dict[str, int]:
    return {k: SIGNAL_TO_NUM.get(str(v).upper(), 0) for k, v in signals.items()}

def generate_signals(
    df: Optional[pd.DataFrame],
    ticker: str,
    avg_sentiment: Optional[dict] = None
) -> Dict[str, str]:
    """Геnotрує all сигнали for тикера"""
    if df is None or df.empty:
        logger.warning(f"[UnifiedSignals] [WARN] DataFrame порожнandй for {ticker}, поверandю HOLD")
        return {sig: "HOLD" for sig in ["forecast_signal",
            "rsi_signal",
            "sentiment_signal",
            "insider_signal",
            "trigger_signal",
            "final_signal"]}

    signals = {}

    # --- Forecast ---
    forecast_thr_bull = get_forecast_threshold(ticker, "bullish")
    forecast_thr_bear = get_forecast_threshold(ticker, "bearish")
    forecast_val = df.get("forecast", pd.Series([0])).iloc[-1]
    signals["forecast_signal"] = "BUY" if forecast_val > forecast_thr_bull else "SELL" if forecast_val < forecast_thr_bear else "HOLD"

    # --- RSI ---
    rsi_val = df.get("RSI_14", pd.Series([50])).iloc[-1]
    low_thr, high_thr = get_rsi_threshold(ticker, "1d")
    signals["rsi_signal"] = "BUY" if rsi_val < low_thr else "SELL" if rsi_val > high_thr else "HOLD"

    # --- Sentiment ---
    sentiment_val = avg_sentiment.get("positive", 0.0) - avg_sentiment.get("negative", 0.0) if avg_sentiment else 0.0
    signals["sentiment_signal"] = numeric_to_signal(sentiment_val, ticker)

    # --- Insider ---
    if "weighted_insider_signal" in df.columns:
        latest_signal = df["weighted_insider_signal"].iloc[-1]
        buy_low, buy_high = get_insider_threshold(ticker, "buy")
        sell_low, sell_high = get_insider_threshold(ticker, "sell")
        if buy_low <= latest_signal <= buy_high:
            signals["insider_signal"] = "BUY"
        elif sell_low <= latest_signal <= sell_high:
            signals["insider_signal"] = "SELL"
        else:
            signals["insider_signal"] = "HOLD"
    else:
        signals["insider_signal"] = "HOLD"

    # --- Triggers ---
    if "mention_spikes" in df.columns and df["mention_spikes"].sum() >= 1:
        signals["trigger_signal"] = "BUY"
    elif "volatility_anomalies" in df.columns and df["volatility_anomalies"].sum() >= 1:
        signals["trigger_signal"] = "SELL"
    else:
        signals["trigger_signal"] = "HOLD"

    # --- Final signal (прandоритет with конфandгу) ---
    for priority in SIGNAL_PRIORITY:
        if signals.get(priority) and signals[priority] != "HOLD":
            signals["final_signal"] = signals[priority]
            break
    else:
        numeric_vals = signals_to_numeric({
            k: v for k, v in signals.items() if k.endswith("_signal") and k not in {"insider_signal", "trigger_signal"}
        })
        avg_score = np.mean(list(numeric_vals.values())) if numeric_vals else 0
        signals["final_signal"] = numeric_to_signal(avg_score, ticker)

    logger.info(f"[UnifiedSignals] [OK] {ticker}  Final={signals['final_signal']}, Signals={signals}")
    return signals