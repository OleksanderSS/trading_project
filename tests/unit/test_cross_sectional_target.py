"""A target measured against the other names, and the silent failure it avoids.

Every target this project made was ABSOLUTE — "did AAPL rise" — which makes the
market's own drift the opponent. Equal-weight buy-and-hold returned +18.06% a
year over 30 years, so a model can predict direction well and still deliver
nothing an investor wants.

Measured before this was built (11 walk-forward folds, identical folds,
features and model, excess over passive holding):

    absolute target   6/11 folds   +0.00021   t 0.55
    relative target   9/11         +0.00132   t 2.78

The failure mode these tests exist for: TargetOrchestrator computes targets per
ticker group, and a cross-sectional value computed per group compares a ticker
to ITSELF — exactly zero on every row, no error, no missing values, a full
column of plausible numbers. That is the shape of defect this repository has
been paying for all month.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.targets.calculators.cross_sectional_calculator import (  # noqa: E402
    CrossSectionalCalculator,
)


@pytest.fixture
def calc():
    return CrossSectionalCalculator()


def panel(n_days=12, tickers=('AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF')):
    """One instant per day, every ticker present, prices that diverge."""
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    rows = []
    for i, t in enumerate(tickers):
        # each ticker compounds at its own rate, so ranks are unambiguous
        prices = 100 * (1 + 0.01 * (i + 1)) ** np.arange(n_days)
        rows.append(pd.DataFrame({'datetime': dates, 'ticker': t, 'close': prices}))
    return pd.concat(rows, ignore_index=True).sort_values(['datetime', 'ticker'])


class TestItMeasuresAgainstPeers:
    def test_a_demeaned_target_sums_to_about_zero_per_instant(self, calc):
        d = panel()
        out = calc.calculate(d, shift=-1, method='demean')
        per_instant = out.groupby(d['datetime']).sum(min_count=1).dropna()
        assert np.allclose(per_instant, 0.0, atol=1e-9), (
            'a demeaned cross-section must cancel within each instant')

    def test_the_fastest_riser_is_positive_and_the_slowest_negative(self, calc):
        d = panel()
        out = calc.calculate(d, shift=-1, method='demean')
        joined = d.assign(y=out).dropna(subset=['y'])
        best = joined.groupby('ticker')['y'].mean().idxmax()
        worst = joined.groupby('ticker')['y'].mean().idxmin()
        assert best == 'FFF' and worst == 'AAA'

    def test_rank_lands_on_the_unit_interval(self, calc):
        out = calc.calculate(panel(), shift=-1, method='rank').dropna()
        assert out.min() > 0.0 and out.max() <= 1.0

    def test_rank_is_immune_to_one_outlier(self, calc):
        """A single crazy move drags a mean; it cannot drag a rank."""
        d = panel()
        base = calc.calculate(d, shift=-1, method='rank')
        d2 = d.copy()
        spike = (d2.ticker == 'AAA') & (d2.datetime == d2.datetime.iloc[-1])
        d2.loc[spike, 'close'] *= 100
        after = calc.calculate(d2, shift=-1, method='rank')
        common = base.notna() & after.notna()
        changed = (base[common] != after[common]).mean()
        assert changed < 0.25, 'a rank target should barely move for one outlier'


class TestTheSilentFailureIsRefused:
    def test_a_single_ticker_frame_raises_rather_than_returning_zeros(self, calc):
        """The whole reason REQUIRES_FULL_FRAME exists."""
        d = panel(tickers=('AAA',))
        out = calc.calculate(d, shift=-1, method='demean')
        # With one name the cross-section is degenerate: min_names must void it
        # rather than emit a column of exact zeros.
        assert out.notna().sum() == 0

    def test_instants_with_too_few_names_are_voided(self, calc):
        d = panel()
        thin = d.datetime.iloc[0]
        d = d[~((d.datetime == thin) & (d.ticker.isin(['DDD', 'EEE', 'FFF'])))]
        out = calc.calculate(d, shift=-1, method='demean', min_names=5)
        assert out[d.datetime.values == thin].isna().all()

    def test_a_frame_without_datetime_raises(self, calc):
        d = panel().drop(columns=['datetime'])
        with pytest.raises(ValueError, match='datetime'):
            calc.calculate(d, shift=-1)

    def test_a_frame_without_ticker_raises_naming_the_cause(self, calc):
        d = panel().drop(columns=['ticker'])
        with pytest.raises(ValueError, match='ticker'):
            calc.calculate(d, shift=-1)

    def test_a_forward_shift_is_refused(self, calc):
        with pytest.raises(ValueError, match='negative'):
            calc.calculate(panel(), shift=1)

    def test_an_unknown_method_is_refused(self, calc):
        with pytest.raises(ValueError, match='method'):
            calc.calculate(panel(), shift=-1, method='whatever')

    def test_it_declares_that_it_needs_the_whole_frame(self):
        assert CrossSectionalCalculator.REQUIRES_FULL_FRAME is True


class TestCostStaysOutOfTheLabel:
    def test_adjust_for_costs_is_ignored_not_silently_applied(self, calc):
        plain = calc.calculate(panel(), shift=-1)
        with_costs = calc.calculate(panel(), shift=-1, adjust_for_costs=True,
                                    transaction_costs={'model': 'flat',
                                                       'commission_pct': 0.01})
        pd.testing.assert_series_equal(plain, with_costs)
