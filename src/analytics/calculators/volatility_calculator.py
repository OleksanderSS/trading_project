"""
Calculates various volatility metrics for financial time series.
This module provides a set of reusable static methods.
"""

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class VolatilityCalculator:
    """A collection of static methods for calculating volatility metrics."""

    @staticmethod
    def calculate_rolling_volatility(
        returns: pd.Series,
        window: int,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculates the annualized rolling volatility
        (standard deviation of returns).

        Args:
            returns (pd.Series): A series of asset returns.
            window (int): The rolling window size.
            periods_per_year (int): Number of trading periods in a year
                                 for annualization.

        Returns:
            pd.Series: The annualized rolling volatility.
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            return pd.Series([], dtype=float)

        rolling_std = returns.rolling(window=window, min_periods=1).std()
        annualized_vol = rolling_std * np.sqrt(periods_per_year)
        return annualized_vol

    @staticmethod
    def calculate_realized_volatility(
        returns: pd.Series,
        window: int,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculates the annualized realized volatility,
        defined as the square root of the sum of squared returns.

        Args:
            returns (pd.Series): A series of asset returns (e.g., intraday).
            window (int): The window over which to sum the squared returns.
            periods_per_year (int): The number of periods in a year
                                 for annualization.

        Returns:
            pd.Series: The annualized realized volatility.
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            return pd.Series([], dtype=float)

        squared_returns = returns**2
        sum_of_squares = squared_returns.rolling(window=window, min_periods=1).sum()

        # Scale the sum of squares to match the total period variance
        annualization_factor = periods_per_year / window
        annualized_variance = sum_of_squares * annualization_factor

        realized_vol = np.sqrt(annualized_variance)
        return realized_vol
