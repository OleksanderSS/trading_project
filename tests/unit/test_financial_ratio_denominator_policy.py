import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
from src.meta_learning.calibration.calibration_engine import CalibrationEngine
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary
from src.monitoring.health_hub import HealthHub
from src.scripts.optimization.portfolio.optimizer import PortfolioOptimizer
from src.pipeline.stages.evaluation.metrics_calculator import MetricsCalculator


def test_risk_reward_sharpe_returns_nan_for_constant_returns():
    result = RiskRewardCalculator.calculate_sharpe_ratio(pd.Series([0.01, 0.01, 0.01]))

    assert np.isnan(result)


def test_risk_reward_sortino_returns_nan_for_zero_downside_std():
    result = RiskRewardCalculator.calculate_sortino_ratio(pd.Series([-0.01, -0.01, 0.02]))

    assert np.isnan(result)


def test_risk_reward_information_ratio_returns_nan_for_identical_series():
    returns = pd.Series([0.01, -0.02, 0.03])

    result = RiskRewardCalculator.calculate_information_ratio(returns, returns.copy())

    assert np.isnan(result)


def test_financial_metrics_library_uses_nan_for_undefined_ratios():
    returns = pd.Series([0.01, 0.01, 0.01])

    assert np.isnan(FinancialMetricsLibrary.calculate_sharpe_ratio(returns))
    assert np.isnan(FinancialMetricsLibrary.calculate_information_ratio(returns, returns.copy()))


def test_portfolio_optimizer_safe_sharpe_returns_nan_for_zero_volatility():
    assert np.isnan(PortfolioOptimizer._safe_sharpe(0.05, 0.0))


def test_inverse_volatility_portfolio_rejects_zero_volatility_inputs():
    optimizer = PortfolioOptimizer()
    returns = pd.DataFrame({"AAA": [0.0, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0]})

    result = optimizer.inverse_volatility_portfolio(returns)

    assert result["success"] is False
    assert "volatility" in result["error"]


def test_diary_metrics_filter_nonfinite_returns():
    diary = object.__new__(DiaryEngine)

    result = diary._calculate_performance_metrics(np.array([1.0, np.nan, np.inf, -1.0]))

    assert result["total_pnl"] == 0.0
    assert result["total_trades"] == 2
    assert np.isfinite(result["sharpe_ratio"])


def test_calibration_sharpe_ignores_nonfinite_returns():
    engine = CalibrationEngine.__new__(CalibrationEngine)
    y_true = np.array([1.0, np.nan, np.inf])
    y_pred = np.array([1.0, 1.0, 1.0])

    assert engine._calculate_sharpe_ratio(y_true, y_pred) == 0.0


def test_metrics_calculator_returns_nan_sharpe_for_zero_volatility():
    calculator = MetricsCalculator()
    portfolio_history = pd.DataFrame({"total_value": [100.0, 100.0, 100.0]})

    result = calculator._calculate_basic_metrics(portfolio_history)

    assert np.isnan(result["sharpe_ratio"])


def test_health_hub_zero_std_drift_uses_mean_delta():
    hub = object.__new__(HealthHub)
    historical = pd.DataFrame({"win_rate": [0.5, 0.5], "sharpe_ratio": [1.0, 1.0]})
    recent = pd.DataFrame({"win_rate": [0.5, 0.5], "sharpe_ratio": [1.2, 1.2]})

    assert hub._calculate_drift_metrics(recent, historical) is True
