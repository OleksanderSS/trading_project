"""
Drawdown Calculator Proxy
Delegates calculation to the unified FinancialMetricsLibrary.
"""
import pandas as pd

from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary


class DrawdownCalculator:
    """Proxy class for drawdown calculations."""

    @staticmethod
    def calculate_max_drawdown_from_returns(returns: pd.Series) -> pd.Series:
        """Proxies to calculate_drawdowns in the library."""
        # Convert returns to equity curve for the library
        equity_curve = (1 + returns).cumprod()
        return FinancialMetricsLibrary.calculate_drawdowns(equity_curve)

    @staticmethod
    def calculate_rolling_drawdown(df: pd.DataFrame, window: int, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """Calculates rolling drawdown using library logic."""
        # Custom logic for high_col vs price_col can be kept or unified
        rolling_max = df[high_col].rolling(window=window, min_periods=1).max()
        return (df[price_col] - rolling_max) / rolling_max

    @staticmethod
    def calculate_max_drawdown_from_prices(df: pd.DataFrame, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """Calculates max drawdown from prices."""
        cumulative_max = df[high_col].cummax()
        return (df[price_col] - cumulative_max) / cumulative_max

    @staticmethod
    def calculate_underwater_duration(df: pd.DataFrame, price_col: str = 'close', high_col: str = 'high') -> pd.Series:
        """Calculates underwater duration."""
        # This one is slightly different from the library's 'max_duration'
        # as it returns a series. We keep the local implementation for Series return.
        cumulative_max = df[high_col].cummax()
        is_underwater = df[price_col] < cumulative_max
        drawdown_blocks = (is_underwater.astype(int).diff().fillna(0) != 0).cumsum()
        return is_underwater.groupby(drawdown_blocks).cumsum()
