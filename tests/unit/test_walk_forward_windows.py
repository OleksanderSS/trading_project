"""Walk-forward windows must be measured in months, not in fractions of input.

The sizes used to be computed from however much history the caller happened to
pass:

    in_sample_size = total_rows * (in / (in + out))     # 80% for a 12/3 config
    step_size      = total_rows * (out / 12)            # assumes 1 year of data

Hand that 30 years of daily bars with the default "12 months in-sample, 3 out"
and it trains on TWENTY-FOUR YEARS and steps forward by SEVEN AND A HALF. The
windows also change size whenever a different amount of history is passed, so
no two runs are comparable — which is worse than being wrong by a constant.

The replacement slices by the data's own calendar, so the same call is correct
for daily, hourly and 15-minute bars.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer  # noqa: E402

windows = WalkForwardOptimizer._walk_forward_windows


def daily(years: float, start='2000-01-03') -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=int(252 * years))
    return pd.DataFrame({'close': 1.0}, index=idx)


def months_between(idx, a, b) -> float:
    return (idx[b - 1] - idx[a]).days / 30.44


class TestWindowsAreMeasuredInMonths:
    def test_in_sample_is_twelve_months_not_a_fraction_of_input(self):
        d = daily(30)
        w = windows(d, 12, 3)
        a, b, _, _ = w[0]
        assert months_between(d.index, a, b) == pytest.approx(12, abs=0.6)

    def test_the_same_config_gives_the_same_window_on_any_history_length(self):
        """The old code's window scaled with the input. This is the whole bug."""
        sizes = []
        for yrs in (10, 20, 30):
            d = daily(yrs)
            a, b, _, _ = windows(d, 12, 3)[0]
            sizes.append(b - a)
        assert max(sizes) - min(sizes) <= 3    # calendar jitter only

    def test_out_of_sample_is_three_months(self):
        d = daily(30)
        _, _, c, e = windows(d, 12, 3)[0]
        assert months_between(d.index, c, e) == pytest.approx(3, abs=0.6)

    def test_thirty_years_yields_many_folds_not_two(self):
        # 30 years at a 3-month step is roughly 116 folds. The old arithmetic
        # produced a handful of enormous ones.
        assert len(windows(daily(30), 12, 3)) > 100


class TestFoldsAreIndependent:
    def test_test_windows_do_not_overlap(self):
        w = windows(daily(20), 12, 3)
        for (_, _, c1, e1), (_, _, c2, _) in zip(w, w[1:]):
            assert e1 <= c2 or c2 >= c1

    def test_training_always_precedes_testing(self):
        for a, b, c, e in windows(daily(20), 12, 3):
            assert a < b <= c < e

    def test_training_window_walks_forward(self):
        w = windows(daily(20), 12, 3)
        assert all(x[0] < y[0] for x, y in zip(w, w[1:]))


class TestItRefusesRatherThanGuesses:
    def test_a_non_datetime_index_yields_no_windows(self):
        # Months cannot be cut out of a RangeIndex, and guessing a
        # bars-per-month figure is what produced the original defect.
        d = pd.DataFrame({'close': [1.0] * 500})
        assert windows(d, 12, 3) == []

    def test_history_shorter_than_one_window_yields_none(self):
        assert windows(daily(0.5), 12, 3) == []

    def test_a_single_row_yields_none(self):
        assert windows(daily(30).head(1), 12, 3) == []

    def test_an_unsorted_index_is_handled_not_trusted(self):
        d = daily(20).sample(frac=1.0, random_state=0)
        assert len(windows(d, 12, 3)) > 10
