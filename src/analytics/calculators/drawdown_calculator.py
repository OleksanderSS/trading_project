"""
Calculates various drawdown and recovery metrics for financial time series.
This module provides a set of reusable static methods.
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class DrawdownCalculator:
    """A collection of static methods to calculate drawdown-related metrics."""

    @staticmethod
    def calculate_max_drawdown_from_returns(returns: pd.Series) -> pd.Series:
        """
        Calculates the drawdown from a series of returns.
        The result is a Series representing the drawdown from the cumulative peak.
        """
        if not isinstance(returns, pd.Series) or returns.empty:
            logger.error("Input for drawdown calculation must be a non-empty pandas Series.")
            return pd.Series([], dtype=float)
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    @staticmethod
    def calculate_rolling_drawdown(df: pd.DataFrame, window: int, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """
        Calculates the rolling drawdown over a specified window.
        """
        if high_col not in df.columns or price_col not in df.columns:
            logger.error(f"Required columns '{high_col}' or '{price_col}' not in DataFrame for rolling drawdown.")
            return pd.Series(np.nan, index=df.index)

        rolling_max = df[high_col].rolling(window=window, min_periods=1).max()
        drawdown = (df[price_col] - rolling_max) / rolling_max
        return drawdown

    @staticmethod
    def calculate_max_drawdown_from_prices(df: pd.DataFrame, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """
        Calculates the maximum drawdown from the all-time high (expanding window) using price data.
        """
        if high_col not in df.columns or price_col not in df.columns:
            logger.error(f"Required columns '{high_col}' or '{price_col}' not in DataFrame for max drawdown.")
            return pd.Series(np.nan, index=df.index)

        cumulative_max = df[high_col].cummax()
        drawdown = (df[price_col] - cumulative_max) / cumulative_max
        return drawdown

    @staticmethod
    def calculate_underwater_duration(df: pd.DataFrame, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """
        Calculates the duration of each drawdown period (time spent "underwater").
        """
        if high_col not in df.columns or price_col not in df.columns:
            logger.error(f"Required columns '{high_col}' or '{price_col}' not in DataFrame for underwater duration.")
            return pd.Series(np.nan, index=df.index)

        cumulative_max = df[high_col].cummax()
        is_underwater = df[price_col] < cumulative_max

        drawdown_blocks = (is_underwater.astype(int).diff().fillna(0) != 0).cumsum()  # audit-ignore: FILLNA_ZERO_SUSPICIOUS
        underwater_duration = is_underwater.groupby(drawdown_blocks).cumsum()
        
        return underwater_duration