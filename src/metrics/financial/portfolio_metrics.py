from typing import Any

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.metrics.base import BaseMetricCalculator

from .financial_metrics_library import FinancialMetricsLibrary


class PortfolioMetricsCalculator(BaseMetricCalculator):
    """
    High-level calculator for portfolio financial metrics.
    Uses FinancialMetricsLibrary for underlying mathematics.
    """

    def __init__(self, config_manager: Any | None = None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("PortfolioMetrics")
        self._trading_days = self.config.get('metrics.trading_days_per_year', 252)
        self._rf_rate = self.config.get('metrics.risk_free_rate', 0.02)

    @property
    def category(self) -> str:
        return "financial"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> dict[str, Any]:
        """Performs full calculation suite for an equity curve."""
        if not self.validate_input(equity_curve):
            return {}

        returns = equity_curve.pct_change().dropna()
        rf = kwargs.get('risk_free_rate', self._rf_rate)
        days = kwargs.get('trading_days', self._trading_days)

        lib = FinancialMetricsLibrary

        return {
            'total_return_pct': lib.calculate_total_return(equity_curve),
            'cagr': lib.calculate_cagr(equity_curve, days),
            'annualized_volatility': lib.calculate_annualized_volatility(returns, days),
            'sharpe_ratio': lib.calculate_sharpe_ratio(returns, rf, days),
            'sortino_ratio': lib.calculate_sortino_ratio(returns, rf, days),
            'max_drawdown': lib.calculate_max_drawdown(equity_curve),
            'calmar_ratio': lib.calculate_calmar_ratio(equity_curve, days),
            'underwater_duration': lib.calculate_underwater_duration(equity_curve)
        }

    def validate_input(self, data: Any) -> bool:
        """Validates that input data is a non-empty pandas Series."""
        return isinstance(data, pd.Series) and not data.empty
