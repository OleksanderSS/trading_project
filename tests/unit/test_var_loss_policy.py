import numpy as np
import pandas as pd
import pytest

from src.analytics.analyzers.risk_decomposition_analyzer import RiskDecompositionAnalyzer
from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator, TradeConfig
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary
from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.risk_management.var_calculator import VaRCalculator


def test_risk_reward_var_cvar_is_loss_positive_with_thresholds():
    returns = pd.Series([0.02, 0.01, -0.05, -0.02, 0.03])

    result = RiskRewardCalculator.calculate_var_cvar(
        returns,
        TradeConfig(confidence_level=0.95),
    )

    assert result["status"] == "ok"
    assert result["var"] > 0
    assert result["cvar"] > 0
    assert result["var_return_threshold"] < 0


def test_var_cvar_empty_returns_are_insufficient_data():
    result = RiskRewardCalculator.calculate_var_cvar(pd.Series(dtype=float))

    assert result["status"] == "insufficient_data"
    assert np.isnan(result["var"])
    assert np.isnan(result["cvar"])


def test_financial_metrics_positive_only_returns_have_zero_tail_loss():
    result = FinancialMetricsLibrary.calculate_var_cvar(
        pd.Series([0.01, 0.02, 0.03, 0.04]),
        confidence_level=0.95,
    )

    assert result["status"] == "ok"
    assert result["var"] == 0.0
    assert result["cvar"] == 0.0
    assert result["var_return_threshold"] > 0


def test_risk_decomposition_reports_loss_positive_var():
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.04, 0.02, -0.01, 0.03, -0.02],
            "BBB": [0.02, -0.03, 0.01, -0.02, 0.02, -0.01],
        }
    )

    result = RiskDecompositionAnalyzer()._calculate_aggregate_risk_profile(
        returns,
        {"AAA": 0.5, "BBB": 0.5},
    )

    assert result["value_at_risk_95"] > 0
    assert result["conditional_var_95"] > 0
    assert result["var_return_threshold_95"] < 0


def test_elite_risk_historical_and_parametric_var_are_loss_positive():
    metrics = EliteRiskMetrics()
    metrics.update_returns(
        "AAA",
        pd.Series([0.01, -0.02, 0.015, -0.03, 0.02, -0.01] * 10),
    )

    historical_var = metrics.compute_historical_simulation_var("AAA")
    parametric_var = metrics.compute_parametric_var("AAA", time_horizon=5)

    assert historical_var > 0
    assert parametric_var["status"] == "ok"
    assert parametric_var["time_horizon"] == 5
    assert parametric_var["var"] >= 0
    assert parametric_var["var_return_threshold"] <= 0


def test_elite_risk_missing_returns_uses_explicit_insufficient_data_fallback():
    metrics = EliteRiskMetrics()

    result = metrics.compute_parametric_var("MISSING")

    assert result["status"] == "insufficient_data"
    assert result["var"] == pytest.approx(metrics.DEFAULT_VAR_LOSS)


def test_var_calculator_empty_and_horizon_policy():
    calculator = VaRCalculator()

    empty_result = calculator.calculate_var_historical([], time_horizon=5)
    loss_result = calculator.calculate_var_historical(
        [0.02, -0.04, 0.01, -0.02],
        time_horizon=4,
    )

    assert empty_result["status"] == "insufficient_data"
    assert np.isnan(empty_result["var"])
    assert loss_result["status"] == "ok"
    assert loss_result["var"] > 0
    assert loss_result["time_horizon"] == 4
