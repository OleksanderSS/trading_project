"""
Modular Adaptive Technical Indicators - Композиційний підхід з розбиттям на модулі
"""

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.utils.data_safety import safe_rolling

logger = ProjectLogger.get_logger("ModularAdaptiveIndicators")


class RSICalculator:
    """Спеціалізований калькулятор RSI"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger("RSICalculator")

    def calculate(self, prices: pd.Series, base_period: int = 14) -> pd.Series:
        """Розрахунок адаптивного RSI з усіма оптимізаціями"""
        try:
            if len(prices.dropna()) == 0:
                return pd.Series([50.0] * len(prices), index=prices.index)
            returns = prices.pct_change(fill_method=None).fillna(0)
            volatility = safe_rolling(returns, window=20, agg="std").fillna(0.01)
            volatility_clean = volatility.fillna(volatility.mean())
            if len(volatility_clean.dropna()) == 0:
                vol_multiplier = pd.Series([1.0] * len(volatility), index=volatility.index)
            else:
                q_90 = volatility_clean.quantile(0.9)
                if pd.isna(q_90):
                    q_90 = volatility_clean.std()
                vol_multiplier = 0.5 + volatility_clean / q_90 * 1.5
                vol_multiplier = vol_multiplier.clip(0.5, 2.0)
            adaptive_period = int(base_period * float(vol_multiplier.mean()))
            adaptive_period = np.clip(adaptive_period, base_period // 2, base_period * 3)
            delta = prices.diff()
            delta_clean = delta.fillna(0)
            gain_simple = safe_rolling(delta_clean.where(delta_clean > 0, 0), window=adaptive_period, agg="mean")
            loss_simple = safe_rolling(-delta_clean.where(delta_clean < 0, 0), window=adaptive_period, agg="mean")
            gain_weighted = (
                delta_clean.where(delta_clean > 0, 0)
                .rolling(window=adaptive_period, win_type="exponential")
                .mean()
                .shift(1)
            )
            loss_weighted = (
                (-delta_clean.where(delta_clean < 0, 0))
                .rolling(window=adaptive_period, win_type="exponential")
                .mean()
                .shift(1)
            )
            rs_simple = gain_simple / (gain_simple + loss_simple).fillna(0.5)
            rs_weighted = gain_weighted / (gain_weighted + loss_weighted).fillna(0.5)
            rs_ultimate = (rs_simple * 0.4 + rs_weighted * 0.6).fillna(0.5)
            rsi = 100 - 100 / (1 + rs_ultimate)
            rsi = rsi.clip(0, 100)
            return rsi.fillna(50.0)
        except Exception as e:
            self.logger.error(f"Error in RSI calculation: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)


class MACDCalculator:
    """Спеціалізований калькулятор MACD"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger("MACDCalculator")

    def calculate(
        self, prices: pd.Series, base_fast: int = 12, base_slow: int = 26, signal_period: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Розрахунок адаптивного MACD з усіма оптимізаціями"""
        try:
            if len(prices.dropna()) == 0:
                empty_series = pd.Series([0.0] * len(prices), index=prices.index)
                return empty_series, empty_series, empty_series
            returns = prices.pct_change(fill_method=None).fillna(0)
            trend_abs = abs(returns.rolling(window=50, min_periods=1).mean()).shift(1).fillna(0.01)
            trend_vol = returns.rolling(window=50, min_periods=1).std().shift(1).fillna(0.01)
            trend_momentum = returns.rolling(window=25, min_periods=1).mean().shift(1).fillna(0.01)
            trend_composite = trend_abs + trend_vol * 0.3 + trend_momentum * 0.2
            fast_factor = 1.0 + np.clip(trend_composite / 0.02, 0.5, 3.0)
            slow_factor = 1.0 + np.clip(trend_composite / 0.01, 0.5, 2.5)
            fast_period = int(base_fast * fast_factor.mean())
            slow_period = int(base_slow * slow_factor.mean())
            fast_period = int(np.clip(fast_period, 5, 50))
            slow_period = int(np.clip(slow_period, 10, 100))
            ema_fast_linear = prices.ewm(span=fast_period).mean()
            ema_slow_linear = prices.ewm(span=slow_period).mean()
            ema_fast_smooth = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow_smooth = prices.ewm(span=slow_period, adjust=False).mean()
            macd_linear = ema_fast_linear - ema_slow_linear
            signal_linear = macd_linear.ewm(span=signal_period).mean()
            macd_linear - signal_linear
            macd_smooth = ema_fast_smooth - ema_slow_smooth
            signal_smooth = macd_smooth.ewm(span=signal_period).mean()
            macd_smooth - signal_smooth
            macd_composite = macd_linear * 0.6 + macd_smooth * 0.4
            signal_composite = signal_linear * 0.6 + signal_smooth * 0.4
            histogram_composite = macd_composite - signal_composite
            if isinstance(macd_composite, pd.Series):
                macd_composite = macd_composite.astype(float)
            if isinstance(signal_composite, pd.Series):
                signal_composite = signal_composite.astype(float)
            if isinstance(histogram_composite, pd.Series):
                histogram_composite = histogram_composite.astype(float)
            return macd_composite.fillna(0), signal_composite.fillna(0), histogram_composite.fillna(0)
        except Exception as e:
            self.logger.error(f"Error in MACD calculation: {e}")
            empty_series = pd.Series([0.0] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series


class BollingerBandsCalculator:
    """Спеціалізований калькулятор Bollinger Bands"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger("BollingerBandsCalculator")

    def calculate(
        self, prices: pd.Series, base_period: int = 20, base_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Розрахунок адаптивних Bollinger Bands з усіма оптимізаціями"""
        try:
            if len(prices.dropna()) == 0:
                mean_price = prices.mean()
                empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
                return empty_series, empty_series, empty_series
            rolling_mean, rolling_std = self._safe_rolling_stats(prices, base_period)
            volatility = rolling_std.fillna(rolling_std.mean())
            vol_range = volatility.max() - volatility.min()
            vol_mean = volatility.mean()
            vol_median = volatility.median()
            if vol_range > 0:
                vol_ratio = (volatility - vol_mean) / vol_range
            else:
                vol_ratio = pd.Series([1.0] * len(volatility), index=volatility.index)
            vol_ratio = vol_ratio.fillna(1.0)
            trend_factor = 1.0 + vol_ratio * 2.0
            range_factor = 1.0 + (volatility - vol_median) / vol_mean if vol_mean > 0 else 1.0
            adaptive_std = base_std * (1 + vol_ratio * trend_factor * range_factor)
            adaptive_std = adaptive_std.clip(base_std * 0.3, base_std * 5.0)
            upper_band = rolling_mean + adaptive_std
            lower_band = rolling_mean - adaptive_std
            if isinstance(upper_band, pd.Series):
                upper_band = upper_band.astype(float)
            if isinstance(lower_band, pd.Series):
                lower_band = lower_band.astype(float)
            if isinstance(rolling_mean, pd.Series):
                rolling_mean = rolling_mean.astype(float)
            return upper_band.fillna(rolling_mean), rolling_mean.fillna(rolling_mean), lower_band.fillna(rolling_mean)
        except Exception as e:
            self.logger.error(f"Error in Bollinger Bands calculation: {e}")
            mean_price = prices.mean()
            empty_series = pd.Series([mean_price] * len(prices), index=prices.index)
            return empty_series, empty_series, empty_series

    def _safe_rolling_stats(self, series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        """Безпечна статистика з обробкою NaN"""
        if len(series.dropna()) == 0:
            mean_val = series.mean()
            empty_series = pd.Series([mean_val] * len(series), index=series.index)
            return empty_series, empty_series
        try:
            rolling_mean = safe_rolling(series, window=window, agg="mean")
            rolling_std = safe_rolling(series, window=window, agg="std")
            mean_filled = rolling_mean.fillna(series.mean())
            std_filled = rolling_std.fillna(series.std())
            return mean_filled, std_filled
        except Exception as e:
            self.logger.error(f"Error in rolling stats: {e}")
            mean_val = series.mean()
            empty_series = pd.Series([mean_val] * len(series), index=series.index)
            return empty_series, empty_series


class ModularAdaptiveTechnicalIndicators:
    """
    Композиційний адаптивний індикатор з розбиттям на модулі
    """

    def __init__(self, config=None):
        self.volatility_window = 20
        self.regime_window = 50
        self.cache = {}
        self.config = config or {}
        self.rsi_calculator = RSICalculator()
        self.macd_calculator = MACDCalculator()
        self.bb_calculator = BollingerBandsCalculator()

    def adaptive_rsi(self, prices: pd.Series, base_period: int = 14) -> pd.Series:
        """Адаптивний RSI з використанням модульного калькулятора"""
        return self.rsi_calculator.calculate(prices, base_period)

    def adaptive_macd(
        self, prices: pd.Series, base_fast: int = 12, base_slow: int = 26, signal_period: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Адаптивний MACD з використанням модульного калькулятора"""
        return self.macd_calculator.calculate(prices, base_fast, base_slow, signal_period)

    def adaptive_bollinger_bands(
        self, prices: pd.Series, base_period: int = 20, base_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Адаптивні Bollinger Bands з використанням модульного калькулятора"""
        return self.bb_calculator.calculate(prices, base_period, base_std)

    def calculate_all_adaptive_indicators(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Розрахунок всіх адаптивних індикаторів"""
        try:
            results = {}
            close = price_data["close"]
            price_data["high"]
            price_data["low"]
            if "volume" not in price_data.columns:
                logger.warning("Volume column not found in price_data. Using default volume.")
                pd.Series([1000.0] * len(price_data), index=price_data.index)
            else:
                price_data["volume"]
            results["adaptive_rsi"] = self.adaptive_rsi(close)
            results["adaptive_macd"] = self.adaptive_macd(close)
            results["adaptive_bollinger_bands"] = self.adaptive_bollinger_bands(close)
            return results
        except Exception as e:
            self.logger.error(f"Виникла помилка: {e}", exc_info=True)
            print(f"Error calculating modular adaptive indicators: {e}")
            return {}

    def get_adaptive_parameters(self, prices: pd.Series) -> dict[str, Any]:
        """Отримання адаптивних параметрів для моніторингу"""
        try:
            returns = prices.pct_change(fill_method=None).fillna(0)
            volatility = safe_rolling(returns, window=self.volatility_window, agg="std").fillna(0.01)
            trend_returns = safe_rolling(returns, window=self.regime_window, agg="mean").fillna(0.01)
            return {
                "current_volatility": float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.01,
                "current_trend": float(abs(trend_returns).iloc[-1])
                if not pd.isna(abs(trend_returns).iloc[-1])
                else 0.01,
                "volatility_regime": "high" if float(volatility.iloc[-1]) > float(volatility.quantile(0.75)) else "low",
                "trend_regime": "trending"
                if float(abs(trend_returns).iloc[-1]) > float(abs(trend_returns).quantile(0.75))
                else "ranging",
            }
        except Exception as e:
            self.logger.error(f"Виникла помилка: {e}", exc_info=True)
            print(f"Error getting adaptive parameters: {e}")
            return {}
