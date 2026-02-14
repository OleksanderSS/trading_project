# core/pipeline/context_features.py

import numpy as np
import pandas as pd
from typing import Optional
from utils.trading_calendar import TradingCalendar
from utils.features import FeatureEngineer


def build_context_features(
    df: pd.DataFrame,
    calendar: Optional[TradingCalendar] = None,
    sma_long: str = "SMA_365",
    sma_short: str = "SMA_30"
) -> pd.DataFrame:
    """
    Побудова контекстних фandчей:
    - фаfor ринку (bull/bear/neutral)
    - уwithгодженandсть тренду
    - макро-фон and волатильнandсть
    - вforємодandя with новинами
    - andнтеграцandя тригерandв у контекст
    """
    df = df.copy()
    context = pd.DataFrame(index=df.index)

    #  Фаfor ринку (череwith FeatureEngineer.detect_market_phase)
    if "close" in df.columns:
        context["market_phase"] = FeatureEngineer().detect_market_phase(df)

    #  Уwithгодженandсть andмпульсу with трендом (череwith RSI or detect_trend_alignment)
    if "RSI_day" in df.columns and "RSI_365" in df.columns:
        context["trend_alignment"] = np.sign(df["RSI_day"].diff()) * np.sign(df["RSI_365"].diff())
    elif "close" in df.columns:
        context["trend_alignment"] = FeatureEngineer().detect_trend_alignment(df)

    #  Макро-баwithовий фон
    if "GDP" in df.columns and "VIX" in df.columns:
        context["macro_bias"] = df["GDP"] - df["VIX"]

    # [DOWN] Волатильнandсть + макро
    if "VIX" in df.columns and "ATR_day" in df.columns:
        context["macro_volatility"] = df["VIX"] + df["ATR_day"]

    # [BRAIN] Сила сигналу with урахуванням новин
    if "momentum_3d" in df.columns and "news_score" in df.columns:
        context["signal_strength"] = df["momentum_3d"] * df["news_score"]

    #  Вага новини with урахуванням фаwithи ринку
    if "market_phase" in context.columns and "news_score" in df.columns:
        phase_weight = context["market_phase"].map({
            "bull": 1.0,
            "bear": 1.2,
            "neutral": 0.8
        }).fillna(1.0)
        context["phase_weighted_score"] = df["news_score"] * phase_weight

    #  Оwithнака торгового дня
    if calendar and "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        context["is_trading_day"] = dates.isin(calendar.trading_dates)

    # [DATA] Модифandкатор сили сигналу
    fe = FeatureEngineer()

    if sma_short in df.columns and sma_long in df.columns:
        alignment = fe.detect_trend_alignment(df)  # 0 or 1
        phase = fe.detect_market_phase(df).map({
            "bull": 1.1,
            "bear": 0.9,
            "neutral": 1.0
        }).fillna(1.0)

        # комбandнований коефandцandєнт: уwithгодженandсть + фаfor ринку
        context["trend_boost_factor"] = (1.0 + alignment * 0.1) * phase
    else:
        context["trend_boost_factor"] = 1.0

    #  Уwithгодженandсть контексту
    if "trend_alignment" in context.columns and "macro_bias" in context.columns:
        context["context_alignment_score"] = (context["trend_alignment"] * context["macro_bias"]).clip(-1, 1)

    #  Інтеграцandя тригерних фandчей
    for col in ["mention_spikes", "sentiment_extremes", "volatility_anomalies"]:
        if col in df.columns:
            context[f"context_{col}"] = df[col].astype(int)

    # [DATA] Уwithгодженandсть новинних тригерandв andwith фаwithою ринку
    if "market_phase" in context.columns and "mention_spikes" in df.columns:
        context["phase_spike_alignment"] = np.where(
            (context["market_phase"] == "bull") & (df["mention_spikes"] > 0), 1,
            np.where((context["market_phase"] == "bear") & (df["mention_spikes"] > 0), -1, 0)
        )

    #  Меand-оwithнаки for вибору цandлand
    context["has_sentiment"] = "news_score" in df.columns
    context["has_macro"] = any(col in df.columns for col in ["GDP", "VIX", "CPI"])
    context["has_impulse"] = any(col in df.columns for col in ["mention_spikes", "sentiment_extremes"])

    return context