"""
Drawdown and Underwater duration metrics for technical price analysis.
"""

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class DrawdownCalculator:
    """
    A collection of static methods to calculate drawdown-related metrics 
    based on technical price action (OHLCV).
    
    NOTE: For portfolio/equity curve drawdown, use PortfolioMetricsCalculator.
    """

    @staticmethod
    def calculate_rolling_drawdown(df: pd.DataFrame, window: int, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """
        Calculates the rolling drawdown over a specified window based on price action.
        """
        if high_col not in df.columns or price_col not in df.columns:
            logger.error(f"Required columns '{high_col}' or '{price_col}' not in DataFrame for rolling drawdown.")
            return pd.Series(np.nan, index=df.index)

        rolling_max = df[high_col].rolling(window=window, min_periods=1).max()
        drawdown = (df[price_col] - rolling_max) / rolling_max
        return drawdown

    @staticmethod
    def calculate_price_based_max_drawdown(df: pd.DataFrame, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """
        Calculates the drawdown from the all-time high (expanding window) using price data.
        
        Useful for assessing the risk of a specific asset price history.
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
        Calculates the duration of each price-based drawdown period (time spent "underwater").
        """
        if high_col not in df.columns or price_col not in df.columns:
            logger.error(f"Required columns '{high_col}' or '{price_col}' not in DataFrame for underwater duration.")
            return pd.Series(np.nan, index=df.index)

        cumulative_max = df[high_col].cummax()
        is_underwater = df[price_col] < cumulative_max

        drawdown_change = is_underwater.astype(int).diff()
        drawdown_blocks = (drawdown_change.where(drawdown_change.notna(), 0) != 0).cumsum()
        underwater_duration = is_underwater.groupby(drawdown_blocks).cumsum()

        return underwater_duration

    @staticmethod
    def calculate_max_drawdown_from_prices(df: pd.DataFrame, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """Alias for calculate_price_based_max_drawdown for backward compatibility."""
        return DrawdownCalculator.calculate_price_based_max_drawdown(df, price_col=price_col, high_col=high_col)

    @staticmethod
    def calculate_max_drawdown_from_returns(returns: pd.Series) -> pd.Series:
        """
        Calculates the drawdown series from a returns series.

        Args:
            returns: Series of periodic returns.

        Returns:
            pd.Series: Drawdown values (negative numbers indicating loss from peak).
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
