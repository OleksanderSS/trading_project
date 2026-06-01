import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from src.core.logging.logger import ProjectLogger
from src.metrics.base import BaseMetricCalculator
from src.config.unified_config_manager import get_current_config

class PortfolioMetricsCalculator(BaseMetricCalculator):
    """
    Калькулятор фінансових метрик портфеля.
    Обчислює показники прибутковості, ризику та просідання.
    """

    def __init__(self, config_manager: Optional[Any] = None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("PortfolioMetrics")
        
        # Отримання параметрів з конфігурації
        self._trading_days_per_year = self.config.get('metrics.trading_days_per_year', 252)
        self._risk_free_rate = self.config.get('metrics.risk_free_rate', 0.02) # Default to 2%

    @property
    def category(self) -> str:
        """Категорія метрик."""
        return "financial"

    def calculate(self, equity_curve: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Виконує повний розрахунок фінансових метрик.
        
        Args:
            equity_curve (pd.Series): Крива капіталу (equity curve).
            **kwargs: Додаткові параметри, які можуть перевизначити конфігурацію.
            
        Returns:
            Dict[str, Any]: Словник з усіма розрахованими метриками.
        """
        if not self.validate_input(equity_curve):
            return {}

        self.logger.info("Початок розрахунку фінансових метрик портфеля...")
        
        returns = equity_curve.pct_change(fill_method=None).dropna()
        
        pnl_metrics = self.calculate_pnl(equity_curve)
        risk_metrics = self.calculate_risk_metrics(returns, **kwargs)
        drawdown_metrics = self.calculate_drawdown(equity_curve)

        all_metrics = {**pnl_metrics, **risk_metrics, **drawdown_metrics}
        
        self.logger.info(f"Розрахунок завершено. Total Return: {all_metrics.get('total_return_pct', 0):.2%}")
        return all_metrics

    def calculate_pnl(self, equity_curve: pd.Series) -> Dict[str, Any]:
        if equity_curve.empty:
            return {'initial_equity': 0.0, 'final_equity': 0.0, 'total_return_pct': 0.0}
        
        initial_equity = equity_curve.iloc[0]
        final_equity = equity_curve.iloc[-1]
        
        total_return = (final_equity - initial_equity) / initial_equity
        
        years = len(equity_curve) / self._trading_days_per_year
        cagr = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0.0
        
        return {
            'initial_equity': float(initial_equity),
            'final_equity': float(final_equity),
            'total_return_pct': float(total_return),
            'cagr': float(cagr)
        }

    def calculate_risk_metrics(self, returns: pd.Series, **kwargs) -> Dict[str, Any]:
        # Дозволяє перевизначати risk_free_rate під час виклику
        risk_free_rate = kwargs.get('risk_free_rate', self._risk_free_rate)
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        annualized_return = (1 + mean_return) ** self._trading_days_per_year - 1 if not np.isnan(mean_return) else 0.0
        annualized_vol = std_return * np.sqrt(self._trading_days_per_year) if not np.isnan(std_return) else 0.0
        
        excess_returns = returns - (risk_free_rate / self._trading_days_per_year)
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self._trading_days_per_year) if excess_returns.std() > 0 else 0.0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self._trading_days_per_year) if not downside_returns.empty and downside_returns.std() > 0 else 0.0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
        
        return {
            'annualized_volatility': float(annualized_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio)
        }

    def calculate_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        rolling_max = equity_curve.expanding(min_periods=1).max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0
        
        is_in_drawdown = drawdowns < 0
        dd_groups = (is_in_drawdown != is_in_drawdown.shift()).cumsum()
        recovery_time_days = 0
        if is_in_drawdown.any():
            drawdown_durations = is_in_drawdown[is_in_drawdown].groupby(dd_groups).size()
            recovery_time_days = int(drawdown_durations.max())
            
        return {
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'recovery_time_days': recovery_time_days
        }

    def validate_input(self, data: Any) -> bool:
        if not isinstance(data, pd.Series) or data.empty:
            self.logger.error("Вхідні дані повинні бути непорожнім pd.Series (equity curve).")
            return False
        return True