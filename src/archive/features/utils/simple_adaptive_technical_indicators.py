"""
Simple Adaptive Technical Indicators - Robust version for real data
"""


import numpy as np
import pandas as pd


class SimpleAdaptiveTechnicalIndicators:
    """
    Simple adaptive technical indicators that handle missing data gracefully.
    """

    def __init__(self):
        self.volatility_window = 20
        self.regime_window = 50

    def adaptive_rsi(self, prices: pd.Series, base_period: int = 14) -> pd.Series:
        """Simple adaptive RSI with causal NaN handling"""
        if len(prices.dropna()) == 0:
            return pd.Series([50.0] * len(prices), index=prices.index)

        # Calculate returns with causal forward fill
        returns = prices.pct_change(fill_method=None).ffill()

        # Simple volatility calculation
        volatility = returns.rolling(self.volatility_window).std().ffill().fillna(0.01)

        # Adaptive period (causal: expanding quantile)
        vol_q90 = volatility.expanding().quantile(0.9).fillna(volatility.std())
        vol_factor = np.clip(volatility / vol_q90, 0.5, 2.0)
        adaptive_period = int(base_period * float(vol_factor.expanding().mean().iloc[-1]))
        adaptive_period = np.clip(adaptive_period, base_period // 2, base_period * 3)

        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=adaptive_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=adaptive_period).mean()
        rs = gain / (gain + loss).fillna(0.5)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50.0)

    def adaptive_macd(self, prices: pd.Series,
                     base_fast: int = 12, base_slow: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Simple adaptive MACD with causal NaN handling"""
        if len(prices.dropna()) == 0:
            empty_series = pd.Series([0.0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

        # Calculate returns with causal forward fill
        returns = prices.pct_change(fill_method=None).ffill()

        # Causal trend detection
        trend = abs(returns.rolling(self.regime_window).mean()).ffill().fillna(0.01)

        # Adaptive periods
        fast_factor = np.clip(trend / 0.02, 0.5, 2.0)
        slow_factor = np.clip(trend / 0.01, 0.5, 2.0)

        fast_period = (base_fast * fast_factor).astype(int).clip(5, 50)
        slow_period = (base_slow * slow_factor).astype(int).clip(10, 100)

        # Calculate MACD (causal EWM)
        ema_fast = pd.concat([prices.ewm(span=p).mean().iloc[i:i+1] for i, p in enumerate(fast_period)])
        ema_slow = pd.concat([prices.ewm(span=p).mean().iloc[i:i+1] for i, p in enumerate(slow_period)])
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return macd_line.ffill().fillna(0), signal_line.ffill().fillna(0), histogram.ffill().fillna(0)

    def adaptive_bollinger_bands(self, prices: pd.Series,
                               base_period: int = 20, base_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Simple adaptive Bollinger Bands with causal NaN handling"""
        if len(prices.dropna()) == 0:
            mean_price = prices.mean()
            empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

        # Calculate rolling statistics
        rolling_mean = prices.rolling(window=base_period).mean().ffill().fillna(prices.mean())
        rolling_std = prices.rolling(window=base_period).std().ffill().fillna(0)

        # Causal adaptive factor using rolling window
        volatility = rolling_std.ffill().fillna(rolling_std.rolling(window=base_period, min_periods=1).mean())
        vol_q90 = volatility.rolling(window=base_period, min_periods=1).quantile(0.9).fillna(volatility.rolling(window=base_period, min_periods=1).std())
        vol_factor = np.clip(volatility / vol_q90, 0.5, 2.0)
        
        adaptive_std = base_std * (1 + vol_factor.rolling(window=base_period, min_periods=1).mean())

        upper_band = rolling_mean + adaptive_std
        lower_band = rolling_mean - adaptive_std

        return upper_band.ffill().fillna(rolling_mean), rolling_mean.ffill().fillna(prices.mean()), lower_band.ffill().fillna(rolling_mean)

    def calculate_all_adaptive_indicators(self, price_data: pd.DataFrame) -> dict:
        """Calculate all simple adaptive indicators"""
        try:
            results = {}

            # Extract series
            close = price_data['close']
            price_data['high']
            price_data['low']
            price_data['volume']

            # Calculate indicators
            results['adaptive_rsi'] = self.adaptive_rsi(close)
            results['adaptive_macd'] = self.adaptive_macd(close)
            results['adaptive_bollinger_bands'] = self.adaptive_bollinger_bands(close)

            return results

        except Exception as e:
            print(f"Error calculating simple adaptive indicators: {e}")
            return {}
