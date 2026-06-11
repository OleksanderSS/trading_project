"""
Hybrid Adaptive Technical Indicators - Best of both worlds
Combines advanced adaptive logic with robust error handling
"""


import numpy as np
import pandas as pd


class HybridAdaptiveTechnicalIndicators:
    """
    Hybrid adaptive technical indicators that combine advanced features with robust error handling.
    """

    def __init__(self):
        self.volatility_window = 20
        self.regime_window = 50

    def adaptive_rsi(self, prices: pd.Series, base_period: int = 14) -> pd.Series:
        """
        Hybrid adaptive RSI with advanced features and robust error handling.
        """
        # Handle empty series
        if len(prices.dropna()) == 0:
            return pd.Series([50.0] * len(prices), index=prices.index)

        try:
            # Advanced volatility calculation with NaN handling
            returns = prices.pct_change(fill_method=None).fillna(0)
            volatility = returns.rolling(self.volatility_window).std().fillna(0.01)

            # Advanced adaptive factor with multiple considerations
            volatility_clean = volatility.fillna(volatility.mean())
            if len(volatility_clean.dropna()) == 0:
                vol_multiplier = pd.Series([1.0] * len(volatility), index=volatility.index)
            else:
                # Quantile-based adaptation with fallback
                q_90 = volatility_clean.quantile(0.9)
                if pd.isna(q_90):
                    q_90 = volatility_clean.std()
                vol_multiplier = 0.5 + (volatility_clean / q_90) * 1.5
                vol_multiplier = vol_multiplier.clip(0.5, 2.0)

            # Adaptive period with bounds checking
            adaptive_period = int(base_period * float(vol_multiplier.mean()))
            adaptive_period = np.clip(adaptive_period, base_period // 2, base_period * 3)

            # Advanced RSI calculation with better handling
            delta = prices.diff()

            # Handle delta with NaN
            delta_clean = delta.fillna(0)

            gain = (delta_clean.where(delta_clean > 0, 0)).rolling(window=adaptive_period).mean()
            loss = (-delta_clean.where(delta_clean < 0, 0)).rolling(window=adaptive_period).mean()

            # Avoid division by zero
            rs_denominator = (gain + loss).fillna(0.5)
            rs = gain / rs_denominator

            # Calculate RSI with bounds
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.clip(0, 100)

            return rsi.fillna(50.0)

        except Exception as e:
            print(f"Error in adaptive_rsi: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)

    def adaptive_macd(self, prices: pd.Series,
                     base_fast: int = 12, base_slow: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Hybrid adaptive MACD with advanced features.
        """
        if len(prices.dropna()) == 0:
            empty_series = pd.Series([0.0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

        try:
            # Advanced returns calculation
            returns = prices.pct_change(fill_method=None).fillna(0)

            # Advanced trend detection with multiple methods
            trend_abs = abs(returns.rolling(self.regime_window).mean()).fillna(0.01)
            trend_vol = returns.rolling(self.regime_window).std().fillna(0.01)

            # Composite trend strength
            trend_strength = trend_abs + (trend_vol * 0.5)

            # Advanced adaptive periods with bounds
            fast_factor = np.clip(trend_strength / 0.02, 0.5, 2.0)
            slow_factor = np.clip(trend_strength / 0.01, 0.5, 2.0)

            fast_period = int(base_fast * fast_factor)
            slow_period = int(base_slow * slow_factor)

            # Apply bounds
            fast_period = np.clip(fast_period, 5, 50)
            slow_period = np.clip(slow_period, 10, 100)

            # Calculate MACD with error handling
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()

            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

            return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)

        except Exception as e:
            print(f"Error in adaptive_macd: {e}")
            empty_series = pd.Series([0.0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def adaptive_bollinger_bands(self, prices: pd.Series,
                               base_period: int = 20, base_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Hybrid adaptive Bollinger Bands with advanced features.
        """
        if len(prices.dropna()) == 0:
            mean_price = prices.mean()
            empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

        try:
            # Advanced rolling statistics with NaN handling
            rolling_mean = prices.rolling(window=base_period).mean()
            rolling_std = prices.rolling(window=base_period).std()

            # Multiple volatility measures
            volatility = rolling_std.fillna(rolling_std.mean())
            vol_range = volatility.max() - volatility.min()
            vol_mean = volatility.mean()

            # Advanced adaptive factor with range consideration
            if vol_range > 0:
                vol_ratio = (volatility - vol_mean) / vol_range
                adaptive_std = base_std * (1 + vol_ratio * 2.0)
            else:
                adaptive_std = pd.Series([base_std] * len(volatility), index=volatility.index)

            # Apply bounds to adaptive std
            adaptive_std = adaptive_std.clip(base_std * 0.5, base_std * 3.0)

            upper_band = rolling_mean + adaptive_std
            lower_band = rolling_mean - adaptive_std

            return upper_band.fillna(rolling_mean), rolling_mean.fillna(rolling_mean), lower_band.fillna(rolling_mean)

        except Exception as e:
            print(f"Error in adaptive_bollinger_bands: {e}")
            mean_price = prices.mean()
            empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def calculate_all_adaptive_indicators(self, price_data: pd.DataFrame) -> dict:
        """
        Calculate all hybrid adaptive indicators.
        """
        try:
            results = {}

            # Extract series
            close = price_data['close']
            price_data['high']
            price_data['low']
            price_data['volume']

            # Calculate all indicators
            results['adaptive_rsi'] = self.adaptive_rsi(close)
            results['adaptive_macd'] = self.adaptive_macd(close)
            results['adaptive_bollinger_bands'] = self.adaptive_bollinger_bands(close)

            return results

        except Exception as e:
            print(f"Error calculating hybrid adaptive indicators: {e}")
            return {}

    def get_adaptive_parameters(self, prices: pd.Series) -> dict:
        """
        Get current adaptive parameters for monitoring.
        """
        try:
            returns = prices.pct_change(fill_method=None).fillna(0)
            volatility = returns.rolling(self.volatility_window).std().fillna(0.01)
            trend_strength = abs(returns.rolling(self.regime_window).mean()).fillna(0.01)

            return {
                'current_volatility': float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.01,
                'current_trend': float(trend_strength.iloc[-1]) if not pd.isna(trend_strength.iloc[-1]) else 0.01,
                'volatility_regime': 'high' if float(volatility.iloc[-1]) > float(volatility.quantile(0.75)) else 'low',
                'trend_regime': 'trending' if float(trend_strength.iloc[-1]) > float(trend_strength.quantile(0.75)) else 'ranging'
            }
        except Exception as e:
            print(f"Error getting adaptive parameters: {e}")
            return {}
