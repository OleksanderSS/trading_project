"""
Volatility Calculator Proxy
Delegates to FinancialMetricsLibrary for unified calculations.
"""
import pandas as pd

from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary


class VolatilityCalculator:
    """Proxy for volatility metrics."""

    @staticmethod
    def calculate_rolling_volatility(returns: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
        """Proxies to the unified library."""
        return FinancialMetricsLibrary.calculate_annualized_volatility(returns, periods_per_year)

    @staticmethod
    def calculate_realized_volatility(returns: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
        """Calculates realized volatility using library-aligned logic."""
        # Custom logic preserved locally for specific realized vol formula
        squared_returns = returns**2
        sum_of_squares = squared_returns.rolling(window=window).sum()
        annualized_variance = sum_of_squares * (periods_per_year / window)
        return annualized_variance**0.5

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Standard ATR calculation (internal tool for RiskReward)."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
