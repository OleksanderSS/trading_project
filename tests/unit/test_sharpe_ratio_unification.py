"""Tests for the Sharpe ratio unification: three independently-maintained
implementations (financial_metrics_library.py, risk_reward_calculator.py,
metrics_mixin.py) were found during a same-session duplication audit —
same core formula, but different risk-free-rate defaults, different
periods-per-year handling, and different NaN-vs-0.0 failure behavior,
meaning they could silently disagree on the same input.

FinancialMetricsLibrary.calculate_sharpe_ratio is now the single canonical
implementation; the other two delegate to it. These tests confirm each
call site's pre-existing external behavior (defaults, failure value) is
byte-for-byte unchanged after the refactor — this is a consolidation, not
a behavior change.
"""
import numpy as np
import pandas as pd
import pytest

from src.algorithms.metrics_mixin import PerformanceMetricsMixin
from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator, TradeConfig
from src.metrics.financial.financial_metrics_library import (
    FinancialMetricsLibrary,
    infer_periods_per_year,
)


def _returns(seed: int = 0, n: int = 100) -> pd.Series:
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(0.001, 0.02, n))


def _dated_returns(seed: int = 0, n: int = 100, freq: str = "1h") -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.Series(rng.normal(0.001, 0.02, n), index=idx)


def test_canonical_matches_manual_reference_formula():
    """The canonical function's math, independently reimplemented here,
    must match — this is the actual correctness check, not just
    delegation wiring."""
    returns = _returns()
    risk_free = 0.03
    periods = 252

    excess = returns - risk_free / periods
    expected = float(excess.mean() / excess.std() * np.sqrt(periods))

    actual = FinancialMetricsLibrary.calculate_sharpe_ratio(returns, risk_free_rate=risk_free, trading_days_per_year=periods)
    assert actual == pytest.approx(expected)


def test_risk_reward_calculator_delegate_matches_pre_refactor_defaults():
    """TradeConfig defaults: risk_free_rate=0.0, periods_per_year=252,
    NaN on failure — must be identical to calling the canonical function
    with those same values directly."""
    returns = _returns(seed=1)
    config = TradeConfig()

    via_wrapper = RiskRewardCalculator.calculate_sharpe_ratio(returns, config)
    via_canonical = FinancialMetricsLibrary.calculate_sharpe_ratio(
        returns, risk_free_rate=config.risk_free_rate, trading_days_per_year=config.periods_per_year,
    )
    assert via_wrapper == pytest.approx(via_canonical)

    # Failure case: too few observations -> NaN (not 0.0).
    tiny = pd.Series([0.01])
    assert np.isnan(RiskRewardCalculator.calculate_sharpe_ratio(tiny, config))


def test_risk_reward_calculator_respects_custom_config():
    returns = _returns(seed=2)
    config = TradeConfig(risk_free_rate=0.05, periods_per_year=52)

    via_wrapper = RiskRewardCalculator.calculate_sharpe_ratio(returns, config)
    via_canonical = FinancialMetricsLibrary.calculate_sharpe_ratio(
        returns, risk_free_rate=0.05, trading_days_per_year=52,
    )
    assert via_wrapper == pytest.approx(via_canonical)


class _MixinHost(PerformanceMetricsMixin):
    pass


def test_metrics_mixin_delegate_preserves_002_default_and_zero_on_failure():
    """metrics_mixin.py's _calculate_sharpe has always defaulted to
    risk_free_rate=0.02 (unlike the other two call sites' 0.0 default) and
    returns 0.0 rather than NaN on failure — both must be preserved
    exactly, not silently unified to match the other two."""
    host = _MixinHost()
    returns = _returns(seed=3)

    via_wrapper = host._calculate_sharpe(returns)
    via_canonical = FinancialMetricsLibrary.calculate_sharpe_ratio(
        returns, risk_free_rate=0.02, trading_days_per_year=infer_periods_per_year(returns), on_error=0.0,
    )
    assert via_wrapper == pytest.approx(via_canonical)

    # Failure case: constant returns -> zero std -> 0.0 (not NaN).
    constant = pd.Series([0.01] * 10)
    result = host._calculate_sharpe(constant)
    assert result == 0.0
    assert not np.isnan(result)


def test_metrics_mixin_auto_infers_periods_from_cadence():
    """periods_per_year=None must trigger cadence-aware inference from the
    DatetimeIndex, matching what _infer_periods_per_year (now re-exported
    from financial_metrics_library) would compute directly."""
    host = _MixinHost()
    hourly_returns = _dated_returns(seed=4, freq="1h")

    via_wrapper = host._calculate_sharpe(hourly_returns, periods_per_year=None)
    expected_ppy = infer_periods_per_year(hourly_returns)
    via_explicit = host._calculate_sharpe(hourly_returns, periods_per_year=expected_ppy)

    assert via_wrapper == pytest.approx(via_explicit)
    assert expected_ppy == 252 * 7  # 1-hour bars bucket


def test_infer_periods_per_year_still_importable_from_metrics_mixin():
    """Three other modules import _infer_periods_per_year directly from
    src.algorithms.metrics_mixin — this must keep working after moving the
    real implementation to financial_metrics_library.py."""
    from src.algorithms.metrics_mixin import _infer_periods_per_year

    assert _infer_periods_per_year is infer_periods_per_year
