"""
Adaptive Technical Indicators - Dynamic indicators that adjust parameters based on market conditions.
Replaces static indicators with volatility-adaptive versions for better signal quality.
"""


import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("AdaptiveTechnicalIndicators")


class AdaptiveTechnicalIndicators:
    """
    Collection of adaptive technical indicators that adjust their parameters
    based on current market volatility and regime conditions.

    Key features:
    - Volatility-adjusted periods
    - Regime-aware parameters
    - Dynamic smoothing factors
    - Market condition adaptation
    """

    def __init__(self):
        self.volatility_window = 20
        self.regime_window = 50

    def adaptive_rsi(self, prices: pd.Series, base_period: int = 14) -> pd.Series:
        """
        Adaptive RSI with dynamic period based on volatility.

        In high volatility: shorter period for faster signals
        In low volatility: longer period for smoother signals
        """
        # Calculate rolling volatility
        returns = prices.pct_change(fill_method=None)
        volatility = returns.rolling(self.volatility_window).std()

        # Normalize volatility to 0.5-2.0 range (multiplier for base period)
        # Handle NaN values in volatility before quantile calculation
        volatility_clean = volatility.fillna(volatility.mean())
        vol_multiplier = 0.5 + (volatility_clean / volatility_clean.quantile(0.9)) * 1.5
        vol_multiplier = vol_multiplier.clip(0.5, 2.0)

        # Calculate adaptive periods with NaN check
        adaptive_periods = (base_period * vol_multiplier).fillna(base_period).astype(int)

        # Calculate RSI with adaptive periods
        rsi_values = []
        for i in range(len(prices)):
            # Check if we have adaptive period for this index
            if i < len(adaptive_periods) and pd.isna(adaptive_periods.iloc[i]):
                rsi_values.append(np.nan)
                continue

            # Get period (adaptive or base)
            period = adaptive_periods.iloc[i] if i < len(adaptive_periods) else base_period

            # Skip if period is invalid
            if pd.isna(period) or period <= 0:
                rsi_values.append(np.nan)
                continue

            # Calculate window safely
            start_idx = max(0, i - int(period) + 1)
            end_idx = i + 1
            window_prices = prices.iloc[start_idx:end_idx]

            # Skip if not enough data
            if len(window_prices) < 2:
                rsi_values.append(np.nan)
                continue

            # Calculate RSI for this window
            delta = window_prices.diff()
            gain = (delta.where(delta > 0, 0)).mean()
            loss = (-delta.where(delta < 0, 0)).mean()

            if loss == 0:
                rsi = 100
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                rsi_values.append(rsi)

        # Ensure rsi_values has same length as prices
        while len(rsi_values) < len(prices):
            rsi_values.append(np.nan)

        return pd.Series(rsi_values, index=prices.index)

    def adaptive_macd(self, prices: pd.Series,
                     base_fast: int = 12, base_slow: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Adaptive MACD with volatility-adjusted periods.

        In trending markets: longer periods for trend confirmation
        In ranging markets: shorter periods for faster signals
        """
        # Calculate market regime (trending vs ranging)
        returns = prices.pct_change(fill_method=None)
        returns_clean = returns.fillna(0)
        trend_strength = abs(returns_clean.rolling(self.regime_window).mean())

        # Normalize trend strength to 0.7-1.3 range
        trend_multiplier = 0.7 + (trend_strength / trend_strength.quantile(0.9)) * 0.6
        trend_multiplier = trend_multiplier.clip(0.7, 1.3)

        # Calculate adaptive periods
        adaptive_fast = (base_fast * trend_multiplier).astype(int)
        adaptive_slow = (base_slow * trend_multiplier).astype(int)

        # Calculate MACD with adaptive periods
        macd_values = []
        signal_values = []

        for i in range(len(prices)):
            if i < adaptive_slow.iloc[i]:
                macd_values.append(np.nan)
                signal_values.append(np.nan)
            else:
                # Calculate EMAs with adaptive periods
                fast_period = adaptive_fast.iloc[i]
                slow_period = adaptive_slow.iloc[i]

                window_prices = prices.iloc[i-slow_period+1:i+1]

                # Simple EMA calculation (can be optimized)
                ema_fast = window_prices.ewm(span=fast_period).mean().iloc[-1]
                ema_slow = window_prices.ewm(span=slow_period).mean().iloc[-1]

                macd = ema_fast - ema_slow
                macd_values.append(macd)

        macd_line = pd.Series(macd_values, index=prices.index)
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def adaptive_bollinger_bands(self, prices: pd.Series,
                               base_period: int = 20, base_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Adaptive Bollinger Bands with volatility-adjusted standard deviation.

        In high volatility: wider bands (higher std multiplier)
        In low volatility: narrower bands (lower std multiplier)
        """
        # Calculate rolling volatility
        returns = prices.pct_change(fill_method=None)
        volatility = returns.rolling(self.volatility_window).std()

        # Adaptive standard deviation multiplier
        # Higher volatility -> wider bands
        vol_ratio = volatility / volatility.rolling(100).mean()  # Relative to historical average
        std_multiplier = base_std * (0.5 + vol_ratio * 1.5)  # 0.5x to 2.0x base std
        std_multiplier = std_multiplier.clip(0.5, 3.0)

        # Calculate adaptive period (shorter in high volatility)
        adaptive_period = (base_period / (1 + volatility)).astype(int)
        adaptive_period = adaptive_period.clip(10, 50)  # Reasonable bounds

        # Calculate bands
        rolling_mean = prices.rolling(adaptive_period).mean()
        rolling_std = prices.rolling(adaptive_period).std()

        upper_band = rolling_mean + (rolling_std * std_multiplier)
        lower_band = rolling_mean - (rolling_std * std_multiplier)

        return upper_band, rolling_mean, lower_band

    def adaptive_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                    base_period: int = 14) -> pd.Series:
        """
        Adaptive ATR with volatility-adjusted period.

        In high volatility: shorter period for faster reaction
        In low volatility: longer period for smoother values
        """
        # Calculate recent volatility
        hl_range = high - low
        volatility = hl_range.rolling(self.volatility_window).std()

        # Adaptive period
        vol_multiplier = 0.7 + (volatility / volatility.quantile(0.9)) * 0.8
        adaptive_period = (base_period / vol_multiplier).astype(int)
        adaptive_period = adaptive_period.clip(7, 30)  # Reasonable bounds

        # Calculate true range
        high_low = high - low
        high_close = np.abs(high.shift(1) - close)
        low_close = np.abs(low.shift(1) - close)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))

        # Adaptive ATR
        atr = true_range.rolling(adaptive_period).mean()

        return atr

    def adaptive_moving_averages(self, prices: pd.Series,
                               base_period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate multiple adaptive moving averages with different characteristics.

        Returns: (EMA, HMA, TEMA, VWAP if volume available)
        """
        # Calculate market conditions
        returns = prices.pct_change(fill_method=None)
        volatility = returns.rolling(self.volatility_window).std()
        abs(returns.rolling(self.regime_window).mean())

        # Adaptive periods
        vol_multiplier = 0.5 + (volatility / volatility.quantile(0.9)) * 1.0
        adaptive_period = (base_period * vol_multiplier).astype(int)
        adaptive_period = adaptive_period.clip(5, 50)

        # 1. Adaptive EMA
        ema = prices.ewm(span=adaptive_period).mean()

        # 2. Adaptive HMA (Hull Moving Average)
        half_period = adaptive_period // 2
        wma_half = self._calculate_wma(prices, half_period)
        wma_full = self._calculate_wma(prices, adaptive_period)
        hma = 2 * wma_half - wma_full

        # 3. Adaptive TEMA (Triple Exponential Moving Average)
        ema1 = prices.ewm(span=adaptive_period).mean()
        ema2 = ema1.ewm(span=adaptive_period).mean()
        ema3 = ema2.ewm(span=adaptive_period).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3

        # 4. Adaptive WMA (Weighted Moving Average)
        wma = self._calculate_wma(prices, adaptive_period)

        return ema, hma, tema, wma

    def adaptive_volume_indicators(self, prices: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate adaptive volume-based indicators.

        Returns: (OBV, VWAP, MFI)
        """
        # Adaptive OBV (On-Balance Volume)
        obv = []
        obv_value = volume.iloc[0] if not pd.isna(volume.iloc[0]) else 0

        for i in range(1, len(prices)):
            if pd.isna(prices.iloc[i]) or pd.isna(prices.iloc[i-1]):
                obv.append(obv_value)
            elif prices.iloc[i] > prices.iloc[i-1]:
                obv_value += volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv_value -= volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0
            # No change in price -> no change in OBV
            obv.append(obv_value)

        obv_series = pd.Series(obv, index=prices.index)

        # Adaptive VWAP (Volume Weighted Average Price)
        # Use adaptive period based on volume volatility
        volume_volatility = volume.pct_change().rolling(self.volatility_window).std()
        vwap_period = (20 * (1 + volume_volatility * 2)).astype(int)
        vwap_period = vwap_period.clip(10, 50)

        vwap = (prices * volume).rolling(vwap_period).sum() / volume.rolling(vwap_period).sum()

        # Adaptive MFI (Money Flow Index)
        typical_price = prices  # Using prices as approximation when high/low are not available
        money_flow = typical_price * volume

        # Adaptive MFI period based on volume volatility
        mfi_period = (14 * (1 + volume_volatility)).astype(int)
        mfi_period = mfi_period.clip(7, 30)

        positive_flow = money_flow.rolling(mfi_period).apply(lambda x: x[x > 0].sum())
        negative_flow = money_flow.rolling(mfi_period).apply(lambda x: x[x < 0].sum())

        mfi = 100 - (100 / (1 + positive_flow / negative_flow))

        return obv_series, vwap, mfi

    def adaptive_momentum_indicators(self, prices: pd.Series,
                                 base_period: int = 10) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate adaptive momentum indicators.

        Returns: (ROC, Momentum, Rate of Acceleration)
        """
        # Calculate market conditions
        returns = prices.pct_change(fill_method=None)
        volatility = returns.rolling(self.volatility_window).std()

        # Adaptive periods
        vol_multiplier = 0.5 + (volatility / volatility.quantile(0.9)) * 1.5
        adaptive_period = (base_period * vol_multiplier).astype(int)
        adaptive_period = adaptive_period.clip(5, 30)

        # 1. Adaptive ROC (Rate of Change)
        roc = ((prices - prices.shift(adaptive_period)) / prices.shift(adaptive_period)) * 100

        # 2. Adaptive Momentum
        momentum = prices - prices.shift(adaptive_period)

        # 3. Rate of Acceleration (second derivative of momentum)
        momentum_roc = momentum.pct_change() * 100

        return roc, momentum, momentum_roc

    def _calculate_wma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return prices.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def calculate_all_adaptive_indicators(self, price_data: pd.DataFrame) -> dict:
        """
        Calculate all adaptive indicators for given price data.

        Args:
            price_data: DataFrame with columns ['close', 'high', 'low', 'volume']

        Returns:
            Dictionary with all adaptive indicators
        """
        try:
            results = {}

            # Extract series
            close = price_data['close']
            high = price_data['high']
            low = price_data['low']
            volume = price_data['volume']

            # Calculate all adaptive indicators
            results['adaptive_rsi'] = self.adaptive_rsi(close)
            results['adaptive_macd'] = self.adaptive_macd(close)
            results['adaptive_bollinger_bands'] = self.adaptive_bollinger_bands(close)
            results['adaptive_atr'] = self.adaptive_atr(high, low, close)
            results['adaptive_moving_averages'] = self.adaptive_moving_averages(close)
            results['adaptive_volume_indicators'] = self.adaptive_volume_indicators(close, volume)
            results['adaptive_momentum_indicators'] = self.adaptive_momentum_indicators(close)
            results['adaptive_parameters'] = self.get_adaptive_parameters(close)

            logger.info(f"Calculated {len(results)} adaptive indicators")
            return results

        except Exception as e:
            logger.error(f"Error calculating adaptive indicators: {e}")
            return {}

    def get_adaptive_parameters(self, prices: pd.Series) -> dict:
        """
        Get current adaptive parameters for monitoring and debugging.
        """
        returns = prices.pct_change(fill_method=None)
        volatility = returns.rolling(self.volatility_window).std()
        trend_strength = abs(returns.rolling(self.regime_window).mean())

        current_vol = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0
        current_trend = trend_strength.iloc[-1] if not pd.isna(trend_strength.iloc[-1]) else 0

        # Calculate multipliers
        vol_multiplier = 0.5 + (current_vol / volatility.quantile(0.9)) * 1.5
        trend_multiplier = 0.7 + (current_trend / trend_strength.quantile(0.9)) * 0.6

        return {
            'current_volatility': current_vol,
            'current_trend_strength': current_trend,
            'volatility_multiplier': vol_multiplier,
            'trend_multiplier': trend_multiplier,
            'market_condition': self._classify_market_condition(current_vol, current_trend)
        }

    def _classify_market_condition(self, volatility: float, trend_strength: float) -> str:
        """Classify current market condition based on volatility and trend."""
        vol_threshold = 0.02  # 2% daily volatility threshold
        trend_threshold = 0.01  # 1% daily trend threshold

        if volatility > vol_threshold:
            if trend_strength > trend_threshold:
                return "high_volatility_trending"
            else:
                return "high_volatility_ranging"
        else:
            if trend_strength > trend_threshold:
                return "low_volatility_trending"
            else:
                return "low_volatility_ranging"
