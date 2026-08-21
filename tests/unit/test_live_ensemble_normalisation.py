"""A missing model quietly shrank every forecast.

`get_weighted_ensemble_prediction` accumulates `weight * value` for the models
that participate, then renormalises the WEIGHTS to sum to one — and leaves the
PREDICTION as the raw sum. When a model drops out (no prediction this bar, or a
weight under the 0.001 floor) the survivors no longer sum to 1, so the returned
number is that fraction of what it should be. A 25% model missing turns every
forecast into 0.75x its own magnitude.

For a regression signal that is a shrinkage toward zero, and KellyCriterion
sizes on magnitude — so positions got smaller whenever a model was
unavailable. An availability accident expressed as reduced conviction.

The second half is the family this project keeps meeting: when nothing
participates the function returns 0.0, which downstream cannot tell from "the
models agree on no move". The empty weights map is the real signal, and the
caller now has to read it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trading.live_adaptive_ensemble import LiveAdaptiveEnsemble  # noqa: E402


@pytest.fixture
def ensemble():
    e = LiveAdaptiveEnsemble.__new__(LiveAdaptiveEnsemble)
    e.logger = logging.getLogger('test-live-ensemble')
    e._weights = {'a': 0.5, 'b': 0.25, 'c': 0.25}
    e.compute_ensemble_weights = lambda regime: dict(e._weights)
    return e


class TestThePredictionIsRenormalisedToo:
    def test_a_full_set_is_the_weighted_average(self, ensemble):
        pred, w = ensemble.get_weighted_ensemble_prediction(
            {'a': 0.10, 'b': 0.10, 'c': 0.10}, 'ranging')
        assert pred == pytest.approx(0.10)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_a_dropped_model_does_not_shrink_the_forecast(self, ensemble):
        """The defect in one test.

        Every survivor says 0.10, so the answer is 0.10 whoever is missing.
        The old code returned 0.75 * 0.10 because 'c' was absent.
        """
        pred, w = ensemble.get_weighted_ensemble_prediction(
            {'a': 0.10, 'b': 0.10}, 'ranging')
        assert pred == pytest.approx(0.10), 'a missing model is not less conviction'
        assert pred != pytest.approx(0.075)

    def test_one_surviving_model_returns_its_own_view(self, ensemble):
        pred, w = ensemble.get_weighted_ensemble_prediction({'b': 0.20}, 'ranging')
        assert pred == pytest.approx(0.20)
        assert w == {'b': pytest.approx(1.0)}

    def test_the_weighted_average_still_respects_weights(self, ensemble):
        # a=0.5, b=0.25 -> (0.5*0.2 + 0.25*0.0) / 0.75
        pred, _ = ensemble.get_weighted_ensemble_prediction(
            {'a': 0.20, 'b': 0.00}, 'ranging')
        assert pred == pytest.approx(0.20 * 0.5 / 0.75)

    def test_sign_is_preserved_through_renormalisation(self, ensemble):
        pred, _ = ensemble.get_weighted_ensemble_prediction(
            {'a': -0.10, 'b': -0.10}, 'ranging')
        assert pred == pytest.approx(-0.10)


class TestAbsenceIsDistinguishable:
    def test_nothing_recognised_returns_empty_weights(self, ensemble):
        pred, w = ensemble.get_weighted_ensemble_prediction(
            {'unknown_model': 0.05}, 'ranging')
        assert w == {}, 'the empty map is the only signal the contract allows'
        assert pred == 0.0

    def test_it_logs_an_error_so_the_reason_is_findable(self, ensemble, caplog):
        with caplog.at_level(logging.ERROR):
            ensemble.get_weighted_ensemble_prediction({'unknown_model': 0.05}, 'ranging')
        assert any('recognised none' in r.message.lower() for r in caplog.records)

    def test_genuine_agreement_on_zero_keeps_its_weights(self, ensemble):
        # The other side: models that DID participate and agree on no move
        # must come back with a non-empty map, so the caller can tell.
        pred, w = ensemble.get_weighted_ensemble_prediction(
            {'a': 0.0, 'b': 0.0}, 'ranging')
        assert pred == 0.0 and w != {}

    def test_a_non_numeric_prediction_is_skipped_not_counted(self, ensemble):
        pred, w = ensemble.get_weighted_ensemble_prediction(
            {'a': 0.10, 'b': 'nonsense'}, 'ranging')
        assert pred == pytest.approx(0.10)
        assert 'b' not in w
