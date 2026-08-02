"""Factor statistics must describe the year they are actually measured over.

analyze_factor_performance carried a SIXTH inline Sharpe,

    (f_series.mean() / factor_std) * np.sqrt(252)

plus `mean * 252` and `std * sqrt(252)` beside it -- all assuming daily
factors. The factor frame inherits the returns index, and this project builds
series from 15m, 60m and 1d bars, so the cadence can be read rather than
assumed.

Worth noting about the scanner that found it: only the ANNUALISATION rule
caught this one. It is an inline expression, not a named function, so the
RIVAL_METRIC rule -- which matches on function names -- looked straight past
it. A useful limit to know.

Currently on a dormant path (hedge_fund_style is disabled in analysis.yaml,
and wrappers.py has no production callers), so this is prevention rather
than repair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.fama_french_factors import FamaFrenchFactors


@pytest.fixture()
def factors():
    return FamaFrenchFactors()


def _series(freq, n=400, seed=0):
    values = np.random.default_rng(seed).normal(0.0004, 0.01, n)
    return pd.DataFrame(
        {"MKT": values},
        index=pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC"),
    )


def test_daily_factors_annualise_over_252_periods(factors):
    stats = factors.analyze_factor_performance(_series("D"))["MKT"]

    assert stats["periods_per_year"] == 252


def test_intraday_factors_annualise_over_more(factors):
    hourly = factors.analyze_factor_performance(_series("h"))["MKT"]
    quarter = factors.analyze_factor_performance(_series("15min"))["MKT"]

    assert quarter["periods_per_year"] > hourly["periods_per_year"] > 252


def test_sharpe_scales_with_the_cadence(factors):
    """The regression: all three used to report the same number."""
    daily = factors.analyze_factor_performance(_series("D"))["MKT"]
    quarter = factors.analyze_factor_performance(_series("15min"))["MKT"]

    assert quarter["annualized_sharpe"] > daily["annualized_sharpe"]


def test_return_and_volatility_use_the_same_year_as_sharpe(factors):
    """Three figures side by side must not describe three different years."""
    stats = factors.analyze_factor_performance(_series("h"))["MKT"]
    periods = stats["periods_per_year"]

    assert stats["annualized_return"] == pytest.approx(
        stats["mean_return"] * periods
    )
    assert stats["annualized_vol"] == pytest.approx(
        stats["volatility"] * np.sqrt(periods)
    )


def test_a_flat_factor_yields_no_sharpe_rather_than_infinity(factors):
    frame = pd.DataFrame(
        {"MKT": np.full(300, 0.001)},
        index=pd.date_range("2026-01-01", periods=300, freq="D", tz="UTC"),
    )

    stats = factors.analyze_factor_performance(frame)["MKT"]

    assert not np.isfinite(stats["annualized_sharpe"]) or stats["annualized_sharpe"] == 0


def test_an_empty_factor_is_skipped(factors):
    frame = pd.DataFrame(
        {"MKT": [np.nan] * 50},
        index=pd.date_range("2026-01-01", periods=50, freq="D", tz="UTC"),
    )

    assert "MKT" not in factors.analyze_factor_performance(frame)


def test_it_delegates_rather_than_recomputing():
    import inspect

    source = inspect.getsource(FamaFrenchFactors.analyze_factor_performance)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "calculate_sharpe_ratio" in code
    assert "np.sqrt(252)" not in code
