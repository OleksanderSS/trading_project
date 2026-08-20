"""Every annualised metric must read the series' own cadence, not assume daily.

This file already contained `infer_periods_per_year`, written with the reason
spelled out: "this project runs 15m/1h/1d timeframes side by side, and a fixed
252 assumes daily bars regardless of what's actually being measured."

It was then applied to `calculate_sharpe_ratio` and to nothing else. CAGR,
annualised volatility, Sortino, Calmar, Treynor and the information ratio all
kept `trading_days_per_year: int = 252` in their signatures and ignored the
inference entirely — so on hourly bars the volatility figure was out by
sqrt(1764/252) ≈ 2.6x, and every ratio built on it inherited that.

A correct function sitting one line away from the callers that need it is this
project's most repeated defect. These tests pin all of them to the inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metrics.financial.financial_metrics_library import (  # noqa: E402
    FinancialMetricsLibrary as F,
    infer_periods_per_year,
)

RNG = np.random.default_rng(0)


def series(freq: str, n: int = 600, loc: float = 0.0005) -> pd.Series:
    idx = pd.date_range('2024-01-02 09:30', periods=n, freq=freq)
    return pd.Series(RNG.normal(loc, 0.01, n), index=idx)


def equity(freq: str, n: int = 600) -> pd.Series:
    return (1 + series(freq, n)).cumprod()


class TestVolatility:
    def test_hourly_is_not_annualised_as_daily(self):
        r = series('h')
        auto = F.calculate_annualized_volatility(r)
        as_daily = F.calculate_annualized_volatility(r, trading_days_per_year=252)
        assert auto != pytest.approx(as_daily, rel=0.01)
        assert auto == pytest.approx(as_daily * np.sqrt(252 * 7 / 252), rel=0.01)

    def test_daily_is_unchanged_by_the_switch(self):
        # The change must not move any number this project has already reported
        # on daily data.
        r = series('B')
        assert F.calculate_annualized_volatility(r) == pytest.approx(
            F.calculate_annualized_volatility(r, trading_days_per_year=252), rel=1e-9)

    def test_an_explicit_factor_still_wins(self):
        r = series('h')
        assert F.calculate_annualized_volatility(r, trading_days_per_year=100) == \
            pytest.approx(float(r.std() * np.sqrt(100)))


class TestEveryOtherAnnualisedMetricInfers:
    @pytest.mark.parametrize('freq,expected', [('B', 252), ('h', 252 * 7)])
    def test_inference_itself_is_right(self, freq, expected):
        assert infer_periods_per_year(series(freq)) == expected

    def test_cagr_differs_between_hourly_and_daily(self):
        e = equity('h')
        assert F.calculate_cagr(e) != pytest.approx(
            F.calculate_cagr(e, trading_days_per_year=252), rel=0.01)

    def test_sortino_differs_between_hourly_and_daily(self):
        r = series('h')
        assert F.calculate_sortino_ratio(r) != pytest.approx(
            F.calculate_sortino_ratio(r, trading_days_per_year=252), rel=0.01)

    def test_treynor_differs_between_hourly_and_daily(self):
        r, b = series('h'), series('h')
        a = F.calculate_treynor_ratio(r, b)
        d = F.calculate_treynor_ratio(r, b, trading_days_per_year=252)
        assert np.isfinite(a) and a != pytest.approx(d, rel=0.01)

    def test_information_ratio_differs_between_hourly_and_daily(self):
        r, b = series('h'), series('h')
        a = F.calculate_information_ratio(r, b)
        d = F.calculate_information_ratio(r, b, trading_days_per_year=252)
        assert np.isfinite(a) and a != pytest.approx(d, rel=0.01)


class TestNoHardcodedDailyRemains:
    def test_no_signature_still_defaults_to_252(self):
        src = Path('src/metrics/financial/financial_metrics_library.py').read_text(encoding='utf-8')
        assert 'int = 252' not in src, (
            'a metric still assumes daily bars in its signature'
        )

    def test_a_non_datetime_index_falls_back_to_daily_not_to_a_crash(self):
        r = pd.Series(RNG.normal(0, 0.01, 300))
        assert np.isfinite(F.calculate_annualized_volatility(r))
        assert infer_periods_per_year(r) == 252
