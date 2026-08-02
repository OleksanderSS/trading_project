# src/features/utils/technical_indicators_lib.py


import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Centralized library for calculating technical indicators."""

    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Calculation of Simple Moving Average (SMA)"""
        return prices.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Calculation of Exponential Moving Average (EMA)"""
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculation of Relative Strength Index (RSI) using Wilder's smoothing"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilder's exponential smoothing (alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculation of Moving Average Convergence Divergence (MACD)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculation of Bollinger Bands"""
        rolling_mean = prices.rolling(period, min_periods=period).mean()
        rolling_std = prices.rolling(period, min_periods=period).std()
        upper_band = rolling_mean + (rolling_std * std)
        lower_band = rolling_mean - (rolling_std * std)
        return upper_band, rolling_mean, lower_band

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculation of Average True Range (ATR)"""
        # True Range compares THIS bar's extremes against the PREVIOUS
        # close: max(H-L, |H - C_prev|, |L - C_prev|). The shift was on the
        # wrong series -- `high.shift(1) - close` is the previous high
        # against the current close, which is not a range anyone defines.
        # On a worked example the two disagree on 3 of 4 bars, in both
        # directions.
        previous_close = close.shift(1)
        high_low = high - low
        high_close = np.abs(high - previous_close)
        low_close = np.abs(low - previous_close)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period, min_periods=period).mean()
        return atr

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        """Calculation of Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculation of Williams %R"""
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r

    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculation of Commodity Channel Index (CCI)"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
        mean_deviation = typical_price.rolling(window=period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
