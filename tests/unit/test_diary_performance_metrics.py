"""Agent comparison metrics, and the fourth Sharpe implementation.

DiaryEngine._calculate_performance_metrics computed

    np.mean(returns) / np.std(returns) * np.sqrt(252)

which is wrong here twice over:

- np.std defaults to the POPULATION deviation (ddof=0); the ratio wants the
  sample deviation;
- sqrt(252) hardcodes a daily cadence, while this diary records 15m, 60m and
  1d decisions. On the same P&L series the annualised figure comes out
  1.144 daily, 3.026 hourly, 5.833 for 15-minute -- the old formula returned
  1.145 for all three, understating intraday by 2.6x and 5.1x.

FinancialMetricsLibrary.calculate_sharpe_ratio is the canonical one, and its
own docstring records that three earlier copies were consolidated into it
after a previous audit found they could silently disagree. This was a
fourth. It now delegates, with the cadence inferred from the timestamps the
diary already stores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.meta_learning.memory.diary_engine import DiaryEngine


@pytest.fixture()
def engine():
    return object.__new__(DiaryEngine)


def _series(freq, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.001, 0.01, n),
        index=pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC"),
    )


def test_the_cadence_changes_the_annualisation(engine):
    """The regression: all three used to return the same number."""
    values = np.random.default_rng(0).normal(0.001, 0.01, 500)

    def sharpe(freq):
        series = pd.Series(
            values,
            index=pd.date_range("2026-01-01", periods=500, freq=freq, tz="UTC"),
        )
        return engine._calculate_performance_metrics(series)["sharpe_ratio"]

    daily, hourly, intraday = sharpe("D"), sharpe("h"), sharpe("15min")

    assert intraday > hourly > daily
    assert daily == pytest.approx(1.14, abs=0.05)


def test_a_series_without_timestamps_falls_back_to_daily(engine):
    metrics = engine._calculate_performance_metrics(np.array([0.01, -0.02, 0.03]))

    assert metrics["total_trades"] == 3
    assert np.isfinite(metrics["sharpe_ratio"])


def test_total_pnl_and_win_rate(engine):
    metrics = engine._calculate_performance_metrics(
        pd.Series([0.02, -0.01, 0.03, -0.005])
    )

    assert metrics["total_pnl"] == pytest.approx(0.035)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["total_trades"] == 4


def test_an_empty_series_yields_zeros_not_an_error(engine):
    metrics = engine._calculate_performance_metrics(pd.Series([], dtype=float))

    assert metrics == {
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "sharpe_ratio": 0.0,
        "total_trades": 0,
    }


def test_infinities_and_nans_are_dropped(engine):
    metrics = engine._calculate_performance_metrics(
        pd.Series([0.01, np.nan, np.inf, -np.inf, 0.02])
    )

    assert metrics["total_trades"] == 2
    assert metrics["total_pnl"] == pytest.approx(0.03)


def test_a_flat_series_has_no_sharpe_rather_than_infinity(engine):
    metrics = engine._calculate_performance_metrics(pd.Series([0.01] * 10))

    assert np.isfinite(metrics["sharpe_ratio"])


def test_it_delegates_instead_of_recomputing():
    import inspect

    source = inspect.getsource(DiaryEngine._calculate_performance_metrics)

    assert "calculate_sharpe_ratio" in source
    assert "np.sqrt(252)" not in source


def test_the_library_agrees_with_the_diary(engine):
    """One implementation means one answer."""
    from src.metrics.financial.financial_metrics_library import (
        FinancialMetricsLibrary,
    )

    series = _series("D")
    direct = FinancialMetricsLibrary.calculate_sharpe_ratio(
        series, trading_days_per_year=None, on_error=0.0
    )

    assert engine._calculate_performance_metrics(series)["sharpe_ratio"] == pytest.approx(
        direct
    )
