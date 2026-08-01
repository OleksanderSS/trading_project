from typing import Any

import pandas as pd

from src.analytics.interfaces import IAnalyzer


class CriticalSignalDetector(IAnalyzer):
    """
    Detects critical market signals such as price shocks, volume spikes,
    and volatility explosions based on dynamic configuration.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initializes the detector with specific configuration settings.

        Args:
            config (Dict[str, Any], optional): A dictionary containing the configuration
                                      for the detector's parameters.
        """
        self.config = config or {}

    def detect_price_shock(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detects a significant price move over a specified window, in either
        direction.

        This used to be `returns < threshold` with a negative threshold, so it
        flagged crashes and ignored an equally violent move up. A melt-up is
        just as much a break from normal conditions, and a caller reading a
        column called `price_shock_detected` has no way to know only half of
        them are in it. The threshold is now read as a MAGNITUDE, so the
        configured -0.07 and a written 0.07 mean the same thing.

        Direction is not discarded: analyze() records it separately, since a
        crash and a spike call for opposite trading responses.
        """
        params = self.config.get('price_shock', {})
        window = params.get('window', 5)
        threshold = abs(params.get('threshold', -0.05))

        if 'close' not in price_data.columns:
            return pd.Series(False, index=price_data.index)

        returns = price_data['close'].pct_change(periods=window, fill_method=None)
        return returns.abs() > threshold

    def price_shock_direction(self, price_data: pd.DataFrame) -> pd.Series:
        """-1 for a shock down, +1 for a shock up, 0 where there is none."""
        params = self.config.get('price_shock', {})
        window = params.get('window', 5)

        if 'close' not in price_data.columns:
            return pd.Series(0, index=price_data.index, dtype='int8')

        returns = price_data['close'].pct_change(periods=window, fill_method=None)
        detected = self.detect_price_shock(price_data)
        direction = pd.Series(0, index=price_data.index, dtype='int8')
        direction[detected & (returns < 0)] = -1
        direction[detected & (returns > 0)] = 1
        return direction

    def detect_volume_spike(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detects unusual spikes in trading volume.
        A volume spike is identified if the current volume exceeds its rolling
        average by a specified multiplier.
        """
        params = self.config.get('volume_spike', {})
        window = params.get('window', 20)
        multiplier = params.get('multiplier', 3.0)

        if 'volume' not in price_data.columns:
            return pd.Series(False, index=price_data.index)

        rolling_avg_volume = price_data['volume'].rolling(window=window).mean()
        return price_data['volume'] > (rolling_avg_volume * multiplier)

    def detect_volatility_explosion(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detects sudden increases in price volatility.
        Volatility is measured as the rolling standard deviation of returns. An
        explosion is identified if the current volatility exceeds its rolling
        average by a specified multiplier.
        """
        params = self.config.get('volatility_explosion', {})
        window = params.get('window', 20)
        multiplier = params.get('multiplier', 2.5)

        if 'close' not in price_data.columns:
            return pd.Series(False, index=price_data.index)

        returns = price_data['close'].pct_change(fill_method=None)
        rolling_volatility = returns.rolling(window=window).std()
        avg_rolling_volatility = rolling_volatility.rolling(window=window).mean()

        return rolling_volatility > (avg_rolling_volatility * multiplier)

    def analyze(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Runs all configured detectors on the input data.

        Args:
            data (pd.DataFrame): Input price data.
            **kwargs: Additional parameters.

        Returns:
            pd.DataFrame: DataFrame with detected signals.
        """
        result_df = data.copy()

        result_df['price_shock_detected'] = self.detect_price_shock(result_df)
        result_df['price_shock_direction'] = self.price_shock_direction(result_df)
        result_df['volume_spike_detected'] = self.detect_volume_spike(result_df)
        result_df['volatility_explosion_detected'] = self.detect_volatility_explosion(result_df)

        return result_df
