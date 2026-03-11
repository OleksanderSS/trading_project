# src/features/utils/technical_indicators_lib.py

import pandas as pd
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    """Централізована бібліотека для розрахунку технічних індикаторів."""

    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Розрахунок Simple Moving Average (SMA)"""
        return prices.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Розрахунок Exponential Moving Average (EMA)"""
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Розрахунок Moving Average Convergence Divergence (MACD)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Розрахунок Bollinger Bands"""
        rolling_mean = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper_band = rolling_mean + (rolling_std * std)
        lower_band = rolling_mean - (rolling_std * std)
        return upper_band, rolling_mean, lower_band

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок Average True Range (ATR)"""
        high_low = high - low
        high_close = np.abs(high.shift(1) - close)
        low_close = np.abs(low.shift(1) - close)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Розрахунок Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r

    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Розрахунок Commodity Channel Index (CCI)"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci