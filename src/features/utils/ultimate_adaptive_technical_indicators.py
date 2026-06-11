"""
Ultimate Adaptive Technical Indicators - The best of all approaches
Combines advanced adaptive logic, robust error handling, and comprehensive features
"""

from typing import Any

import numpy as np
import pandas as pd


class UltimateAdaptiveTechnicalIndicators:
    """
    Ultimate adaptive technical indicators with the best features from all approaches.
    """

    def __init__(self):
        self.volatility_window = 20
        self.regime_window = 50
        self.cache = {}  # Simple cache for performance

    def _safe_rolling_stats(self, series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        """Safe rolling statistics with comprehensive NaN handling"""
        if len(series.dropna()) == 0:
            mean_val = series.mean()
            empty_series = pd.Series([mean_val] * len(series), index=series.index)
            return empty_series, empty_series

        try:
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()

            # Handle all NaN cases
            mean_filled = rolling_mean.fillna(series.mean())
            std_filled = rolling_std.fillna(series.std())

            return mean_filled, std_filled

        except Exception as e:
            print(f"Error in rolling stats: {e}")
            mean_val = series.mean()
            empty_series = pd.Series([mean_val] * len(series), index=series.index)
            return empty_series, empty_series

    def adaptive_rsi(self, prices: pd.Series, base_period: int = 14) -> pd.Series:
        """
        Ultimate adaptive RSI with all advanced features.
        """
        # Handle empty series
        if len(prices.dropna()) == 0:
            return pd.Series([50.0] * len(prices), index=prices.index)

        try:
            # Advanced returns calculation
            returns = prices.pct_change(fill_method=None).fillna(0)

            # Multi-dimensional volatility analysis
            volatility = returns.rolling(self.volatility_window).std().fillna(0.01)
            volatility_rank = volatility.rolling(self.volatility_window).rank(pct=True)
            volatility.rolling(self.volatility_window).quantile([0.25, 0.5, 0.75, 0.9])

            # Advanced adaptive period calculation
            vol_ratio = (volatility - volatility.quantile(0.5)) / (volatility.quantile(0.9) - volatility.quantile(0.5))
            vol_ratio = vol_ratio.fillna(1.0)

            # Multiple adaptive factors
            trend_factor = 1.0 + vol_ratio * 1.5
            regime_factor = 1.0 + (volatility_rank / len(volatility)) * 0.5

            adaptive_period = int(base_period * trend_factor * regime_factor)
            adaptive_period = np.clip(adaptive_period, base_period // 3, base_period * 3)

            # Enhanced RSI calculation
            delta = prices.diff()
            delta_clean = delta.fillna(0)

            # Multiple gain/loss calculations
            gain_simple = (delta_clean.where(delta_clean > 0, 0)).rolling(window=adaptive_period).mean()
            loss_simple = (-delta_clean.where(delta_clean < 0, 0)).rolling(window=adaptive_period).mean()

            gain_weighted = (delta_clean.where(delta_clean > 0, 0)).rolling(window=adaptive_period, win_type='exponential').mean()
            loss_weighted = (-delta_clean.where(delta_clean < 0, 0)).rolling(window=adaptive_period, win_type='exponential').mean()

            # Composite RSI
            rs_simple = gain_simple / (gain_simple + loss_simple).fillna(0.5)
            rs_weighted = gain_weighted / (gain_weighted + loss_weighted).fillna(0.5)

            # Weighted average of multiple RSIs
            rs_ultimate = (rs_simple * 0.4 + rs_weighted * 0.6).fillna(0.5)

            # Calculate RSI with bounds
            rsi = 100 - (100 / (1 + rs_ultimate))
            rsi = rsi.clip(0, 100)

            return rsi.fillna(50.0)

        except Exception as e:
            print(f"Error in ultimate adaptive RSI: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)

    def adaptive_macd(self, prices: pd.Series,
                     base_fast: int = 12, base_slow: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Ultimate adaptive MACD with all advanced features.
        """
        if len(prices.dropna()) == 0:
            empty_series = pd.Series([0.0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

        try:
            # Advanced returns analysis
            returns = prices.pct_change(fill_method=None).fillna(0)

            # Multiple trend detection methods
            trend_abs = abs(returns.rolling(self.regime_window).mean()).fillna(0.01)
            trend_vol = returns.rolling(self.regime_window).std().fillna(0.01)
            trend_momentum = returns.rolling(self.regime_window // 2).mean().fillna(0.01)

            # Composite trend strength
            trend_composite = (trend_abs + trend_vol * 0.3 + trend_momentum * 0.2)

            # Advanced adaptive periods with multiple factors
            fast_factor = 1.0 + np.clip(trend_composite / 0.02, 0.3, 3.0)
            slow_factor = 1.0 + np.clip(trend_composite / 0.01, 0.5, 2.5)

            fast_period = int(base_fast * fast_factor)
            slow_period = int(base_slow * slow_factor)

            # Apply bounds
            fast_period = np.clip(fast_period, 5, 50)
            slow_period = np.clip(slow_period, 10, 100)

            # Advanced MACD calculation with multiple EMA types
            ema_fast_linear = prices.ewm(span=fast_period).mean()
            ema_slow_linear = prices.ewm(span=slow_period).mean()
            ema_fast_smooth = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow_smooth = prices.ewm(span=slow_period, adjust=False).mean()

            # Linear MACD
            macd_linear = ema_fast_linear - ema_slow_linear
            signal_linear = macd_linear.ewm(span=signal_period).mean()
            macd_linear - signal_linear

            # Smooth MACD
            macd_smooth = ema_fast_smooth - ema_slow_smooth
            signal_smooth = macd_smooth.ewm(span=signal_period).mean()
            macd_smooth - signal_smooth

            # Composite MACD (weighted average)
            macd_composite = (macd_linear * 0.6 + macd_smooth * 0.4)
            signal_composite = (signal_linear * 0.6 + signal_smooth * 0.4)
            histogram_composite = macd_composite - signal_composite

            return macd_composite.fillna(0), signal_composite.fillna(0), histogram_composite.fillna(0)

        except Exception as e:
            print(f"Error in ultimate adaptive MACD: {e}")
            empty_series = pd.Series([0.0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def adaptive_bollinger_bands(self, prices: pd.Series,
                               base_period: int = 20, base_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Ultimate adaptive Bollinger Bands with all advanced features.
        """
        if len(prices.dropna()) == 0:
            mean_price = prices.mean()
            empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

        try:
            # Advanced rolling statistics
            rolling_mean, rolling_std = self._safe_rolling_stats(prices, base_period)

            # Multiple volatility measures
            volatility = rolling_std.fillna(rolling_std.mean())
            vol_range = volatility.max() - volatility.min()
            vol_mean = volatility.mean()
            vol_median = volatility.median()

            # Advanced adaptive factor with multiple considerations
            vol_ratio = (volatility - vol_mean) / vol_range if vol_range > 0 else pd.Series(1.0, index=volatility.index)
            vol_ratio = vol_ratio.fillna(1.0)

            # Multi-factor adaptive std
            trend_factor = 1.0 + vol_ratio * 2.0
            range_factor = 1.0 + (volatility - vol_median) / vol_mean if vol_mean > 0 else 1.0

            adaptive_std = base_std * (1 + vol_ratio * trend_factor * range_factor)

            # Apply bounds
            adaptive_std = adaptive_std.clip(base_std * 0.3, base_std * 5.0)

            upper_band = rolling_mean + adaptive_std
            lower_band = rolling_mean - adaptive_std

            return upper_band.fillna(rolling_mean), rolling_mean.fillna(rolling_mean), lower_band.fillna(rolling_mean)

        except Exception as e:
            print(f"Error in ultimate adaptive Bollinger Bands: {e}")
            mean_price = prices.mean()
            empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def calculate_all_adaptive_indicators(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate all ultimate adaptive indicators.
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

            # Add adaptive parameters
            results['adaptive_parameters'] = self.get_adaptive_parameters(close)

            return results

        except Exception as e:
            print(f"Error calculating ultimate adaptive indicators: {e}")
            return {}

    def get_adaptive_parameters(self, prices: pd.Series) -> dict[str, Any]:
        """
        Get comprehensive adaptive parameters for monitoring.
        """
        try:
            if len(prices.dropna()) == 0:
                return {
                    'current_volatility': 0.01,
                    'current_trend': 0.01,
                    'volatility_regime': 'low',
                    'trend_regime': 'ranging',
                    'volatility_percentiles': {'25': 0.01, '50': 0.01, '75': 0.01, '90': 0.01},
                    'trend_percentiles': {'25': 0.01, '50': 0.01, '75': 0.01, '90': 0.01}
                }

            returns = prices.pct_change(fill_method=None).fillna(0)
            volatility = returns.rolling(self.volatility_window).std().fillna(0.01)
            trend_strength = abs(returns.rolling(self.regime_window).mean()).fillna(0.01)

            return {
                'current_volatility': float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.01,
                'current_trend': float(trend_strength.iloc[-1]) if not pd.isna(trend_strength.iloc[-1]) else 0.01,
                'volatility_regime': 'high' if float(volatility.iloc[-1]) > float(volatility.quantile(0.75)) else 'low',
                'trend_regime': 'trending' if float(trend_strength.iloc[-1]) > float(trend_strength.quantile(0.75)) else 'ranging',
                'volatility_percentiles': {
                    '25': float(volatility.quantile(0.25)),
                    '50': float(volatility.quantile(0.5)),
                    '75': float(volatility.quantile(0.75)),
                    '90': float(volatility.quantile(0.9))
                },
                'trend_percentiles': {
                    '25': float(trend_strength.quantile(0.25)),
                    '50': float(trend_strength.quantile(0.5)),
                    '75': float(trend_strength.quantile(0.75)),
                    '90': float(trend_strength.quantile(0.9))
                }
            }

        except Exception as e:
            print(f"Error getting adaptive parameters: {e}")
            return {}
