"""A model has to beat owning the same thing and doing nothing.

Added because the absence of this one number made a worthless result read as a
triumph. A walk-forward over eleven independent folds scored positive in
ELEVEN OF ELEVEN, which looks like success until the passive baseline sits
beside it:

    absolute target   11/11 folds positive   excess over passive +0.00021, t=0.55
    relative target    9/11 folds positive   excess over passive +0.00132, t=2.78

The first arm was earning the market and adding noise. The gate's existing
opponents cannot see that: a constant predictor and a persistence predictor are
both about the SHAPE of the series, and neither asks the question an investor
asks.

Most of these tests are ways the comparison must REFUSE to produce a number.
An invented benchmark is worse than an absent one, because it gates on
something meaningless and nobody can tell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.base_trainer import BaseTrainer  # noqa: E402

score = BaseTrainer._score_passive_holding
RNG = np.random.default_rng(0)


def returns(n=200, loc=0.0, scale=0.01):
    return RNG.normal(loc, scale, n)


class TestItMeasuresWhatItClaims:
    def test_a_model_that_picks_the_best_rows_shows_positive_excess(self):
        y = returns()
        out = score(y, y, is_classif=False)          # perfect foresight
        assert out['passive_status'] == 'measured'
        assert out['excess_over_passive'] > 0
        assert out['selected_mean'] > out['passive_mean']

    def test_a_model_that_picks_the_worst_rows_shows_negative_excess(self):
        y = returns()
        assert score(y, -y, is_classif=False)['excess_over_passive'] < 0

    def test_a_useless_model_lands_near_zero(self):
        y = returns()
        out = score(y, RNG.normal(0, 1, len(y)), is_classif=False)
        assert abs(out['excess_over_passive']) < 0.005

    def test_a_rising_market_does_not_by_itself_create_excess(self):
        # The trap this exists for: every row positive, so any selection looks
        # profitable while adding nothing.
        y = np.abs(returns(loc=0.02)) + 0.001
        out = score(y, RNG.normal(0, 1, len(y)), is_classif=False)
        if out['passive_status'] == 'measured':
            assert abs(out['excess_over_passive']) < 0.01
        # a one-signed series is also allowed to refuse — see the next class

    def test_it_selects_the_configured_share(self):
        y = returns(n=100)
        out = score(y, np.arange(100.0), is_classif=False)
        assert out['selected_count'] == pytest.approx(30, abs=2)


class TestItRefusesRatherThanInventsANumber:
    def test_a_classification_target_has_no_payoff(self):
        out = score(np.array([0, 1] * 50), np.array([0, 1] * 50), is_classif=True)
        assert out['excess_over_passive'] is None
        assert out['passive_status'] == 'not_applicable_classification'

    def test_a_price_or_volume_target_is_not_comparable_to_holding(self):
        # Values in the hundreds are not returns; benchmarking them against
        # "holding everything" would be meaningless.
        out = score(RNG.normal(250, 5, 200), RNG.normal(0, 1, 200), is_classif=False)
        assert out['excess_over_passive'] is None
        assert out['passive_status'] == 'not_a_return_target'

    def test_a_one_signed_series_is_not_treated_as_returns(self):
        out = score(np.abs(returns()) + 0.01, RNG.normal(0, 1, 200), is_classif=False)
        assert out['passive_status'] == 'not_a_return_target'

    def test_constant_predictions_cannot_define_a_top_share(self):
        out = score(returns(), np.full(200, 0.5), is_classif=False)
        assert out['excess_over_passive'] is None
        assert out['passive_status'] == 'predictions_are_constant'

    def test_too_few_rows_refuses(self):
        assert score(returns(5), returns(5), is_classif=False)['excess_over_passive'] is None

    def test_a_length_mismatch_refuses(self):
        out = score(returns(200), returns(50), is_classif=False)
        assert out['passive_status'] == 'length_mismatch'

    def test_unreadable_input_refuses_rather_than_raising(self):
        out = score(['a', 'b', 'c'] * 50, returns(150), is_classif=False)
        assert out['excess_over_passive'] is None

    def test_non_finite_rows_are_dropped_not_propagated(self):
        y = returns(); y[:20] = np.nan
        out = score(y, y, is_classif=False)
        assert out['passive_status'] == 'measured'
        assert np.isfinite(out['excess_over_passive'])


class TestRiskMatchedComparison:
    """Beating passive on raw return is not enough, and leverage is why.

    Measured 2026-08-20 on a real 27-year overlapping portfolio: a
    cross-sectional model beat passive equal weight in 20 years of 28 with a
    median excess of +7.15%, and every per-trade figure agreed there was an
    edge. Volatility was 1.35x and return 1.31x — the same holding levered to
    the same volatility returns +23.14% against the strategy's +23.62%, so the
    model contributed +0.48% a year and a drawdown of -68% against -51%.

    Concentrating into a top share raises risk mechanically. A gate reading
    only the mean promotes that and calls it skill.
    """

    def test_a_pure_leverage_model_shows_no_risk_adjusted_excess(self):
        # Selection that simply picks the most volatile rows: raw mean rises,
        # and it must show up as ~nothing once risk is matched.
        base = RNG.normal(0.001, 0.01, 400)
        y = np.concatenate([base, base * 2.0])          # same edge, twice the risk
        preds = np.concatenate([np.zeros(400), np.ones(400)])
        out = score(y, preds, is_classif=False)
        assert out['excess_over_passive'] > 0, 'raw excess is what fools the gate'
        assert abs(out['excess_risk_adjusted']) < abs(out['excess_over_passive'])

    def test_genuine_selection_survives_the_risk_match(self):
        y = RNG.normal(0, 0.01, 600)
        y[:180] += 0.02                                  # a real, low-variance edge
        preds = np.zeros(600); preds[:180] = 1.0
        out = score(y, preds, is_classif=False)
        assert out['excess_risk_adjusted'] > 0

    def test_the_matched_benchmark_scales_with_the_selection_risk(self):
        y = RNG.normal(0.001, 0.01, 500)
        out = score(y, RNG.normal(0, 1, 500), is_classif=False)
        ratio = out['selected_std'] / out['passive_std']
        assert out['risk_matched_passive'] == pytest.approx(
            out['passive_mean'] * ratio, rel=1e-9)

    def test_it_reports_both_numbers_so_neither_can_hide(self):
        out = score(RNG.normal(0.001, 0.01, 400), RNG.normal(0, 1, 400), is_classif=False)
        for key in ('excess_over_passive', 'excess_risk_adjusted',
                    'passive_std', 'selected_std', 'risk_matched_passive'):
            assert key in out
