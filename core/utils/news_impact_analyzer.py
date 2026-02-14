# core/utils/news_impact_analyzer.py

import pandas as pd
import numpy as np


class NewsImpactAnalyzer:
    def __init__(self, price_df, market_df=None):
        """
        :param price_df: DataFrame with OHLCV for тикера (15m, 60m, 1d)
        :param market_df: DataFrame with ринковим контекстом (SPY, VIX тощо)
        """
        self.price_df = price_df
        self.market_df = market_df

    def initial_signal(self, news_time, ticker, window="15min"):
        """
        Початковий andмпульс на найближчandй свandчцand пandсля новини.
        """
        df = self.price_df[self.price_df["ticker"] == ticker].copy()
        df = df.set_index("date").sort_index()

        # withнайти найближчу свandчку пandсля новини
        news_time = pd.to_datetime(news_time)
        df.index = pd.to_datetime(df.index)
        nearest = df[df.index >= news_time].iloc[0]

        # withмandна цandни вandд попереднього close
        prev_close = df[df.index < news_time].iloc[-1]["close"]
        delta_p = nearest["close"] - prev_close

        # середня волатильнandсть (ATR proxy)
        sigma = df["close"].pct_change().std()

        s0 = abs(delta_p) / sigma if sigma > 0 else 0
        return {"initial_strength": s0, "delta_price": delta_p}

    def short_term_behavior(self, news_time, ticker, n_candles=4):
        """
        Аналandwith кandлькох наступних 15m candles.
        """
        df = self.price_df[self.price_df["ticker"] == ticker].copy()
        df = df.set_index("date").sort_index()

        news_time = pd.to_datetime(news_time)
        df.index = pd.to_datetime(df.index)
        after_news = df[df.index >= news_time].iloc[:n_candles]
        returns = after_news["close"].pct_change().fillna(0)

        # класифandкацandя патерну
        if (returns < 0).sum() >= 3:
            reaction = "cascade_down"
        elif (returns > 0).sum() >= 3:
            reaction = "rebound_up"
        elif returns.abs().mean() < 0.002:
            reaction = "stagnation"
        else:
            reaction = "mixed"

        return {"reaction_type": reaction, "avg_return": returns.mean()}

    def market_context_adjust(self, signal_strength, news_time):
        """
        Корекцandя сили сигналу for ринковим контекстом (SPY, VIX).
        """
        if self.market_df is None:
            return signal_strength

        spy = self.market_df[self.market_df["ticker"] == "SPY"].set_index("date")
        vix = self.market_df[self.market_df["ticker"] == "VIX"].set_index("date")
        
        # Конвертуємо andнwhereкси в datetime
        news_time = pd.to_datetime(news_time)
        spy.index = pd.to_datetime(spy.index)
        vix.index = pd.to_datetime(vix.index)

        # беремо найближчand values
        spy_ret = spy.loc[:news_time].iloc[-1]["close"] / spy.iloc[-2]["close"] - 1
        vix_val = vix.loc[:news_time].iloc[-1]["close"]

        # простand коефandцandєнти
        if spy_ret < 0 and vix_val > vix["close"].mean():
            signal_strength *= 1.2  # посилення при risk-off
        elif spy_ret > 0:
            signal_strength *= 0.8  # послаблення при risk-on

        return signal_strength

    def cascade_to_timeframes(self, s0, reaction_type):
        """
        Переnotсення сили сигналу на довшand andймфрейми.
        """
        coeffs = {"15m": 1.0, "60m": 0.65, "1d": 0.35}

        if reaction_type == "cascade_down":
            coeffs = {tf: c * 1.2 for tf, c in coeffs.items()}
        elif reaction_type == "rebound_up":
            coeffs = {tf: c * 0.5 for tf, c in coeffs.items()}
        elif reaction_type == "stagnation":
            coeffs = {tf: c * 0.8 for tf, c in coeffs.items()}

        return {tf: s0 * coeffs[tf] for tf in coeffs}

    def analyze_news(self, news_time, ticker):
        """
        Повний аналandwith новини.
        """
        init = self.initial_signal(news_time, ticker)
        behavior = self.short_term_behavior(news_time, ticker)
        adjusted = self.market_context_adjust(init["initial_strength"], news_time)
        cascaded = self.cascade_to_timeframes(adjusted, behavior["reaction_type"])

        return {
            "initial": init,
            "behavior": behavior,
            "adjusted_strength": adjusted,
            "cascaded_strength": cascaded
        }