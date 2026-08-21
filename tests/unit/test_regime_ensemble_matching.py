"""A silenced ensemble looked exactly like a calm one, forever.

`generate_weighted_ensemble` looks each prediction up in `regime_weights`,
which is keyed by bare architecture names. `predictions_dict` is keyed by the
model ids the system actually builds -- 'LGBM_5m', 'CatBoost_AAPL_1d'. Equality
matching therefore missed everything, every weight was 0.0, and the function
returned an ensemble score of exactly 0.0.

Downstream reads 0.0 as "no move" and issues HOLD. So an engine that
recognised nothing was indistinguishable from an engine whose models agreed
there was nothing to do -- in every regime-aware call it has ever made.

Three of the five architectures it lists (transformer, lstm, cnn) were moved to
the archive and are not produced at all any more, so even exact matching could
not have saved it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trading.consensus_engine import EnhancedConsensusEngine  # noqa: E402

WEIGHTS = {'linear': 0.30, 'catboost': 0.25, 'lightgbm': 0.25, 'transformer': 0.20}


@pytest.fixture
def engine():
    e = EnhancedConsensusEngine.__new__(EnhancedConsensusEngine)
    e.logger = logging.getLogger('test-consensus')
    e.regime_weights = {'ranging': dict(WEIGHTS), 'trending_up': dict(WEIGHTS)}
    return e


class TestRealModelIdsAreRecognised:
    @pytest.mark.parametrize('model_id,expected', [
        ('LGBM_5m', 'lightgbm'),
        ('CatBoost_AAPL_1d', 'catboost'),
        ('Transformer_v1', 'transformer'),
        ('linear_regression_60m', 'linear'),
        ('lightgbm', 'lightgbm'),
    ])
    def test_an_id_maps_to_its_architecture(self, model_id, expected):
        assert EnhancedConsensusEngine._architecture_of(model_id, WEIGHTS) == expected

    def test_an_unknown_architecture_maps_to_nothing(self):
        assert EnhancedConsensusEngine._architecture_of('prophet_v2', WEIGHTS) is None

    def test_an_empty_name_does_not_crash(self):
        assert EnhancedConsensusEngine._architecture_of('', WEIGHTS) is None

    def test_a_real_prediction_set_now_produces_a_score(self, engine):
        out = engine.generate_weighted_ensemble(
            {'LGBM_5m': 0.02, 'CatBoost_AAPL_1d': 0.04}, 'ranging')
        assert out['ensemble_prediction'] is not None
        assert out['ensemble_prediction'] == pytest.approx(0.03, abs=1e-9)


class TestNothingRecognisedIsNotZero:
    def test_it_returns_none_rather_than_a_number(self, engine):
        """The defect in one test: 0.0 is a decision, None is an absence."""
        out = engine.generate_weighted_ensemble(
            {'prophet_v2': 0.05, 'arima_1': -0.03}, 'ranging')
        assert out['ensemble_prediction'] is None
        assert out['status'] == 'no_recognised_architecture'

    def test_it_names_what_it_could_not_match(self, engine):
        out = engine.generate_weighted_ensemble({'prophet_v2': 0.05}, 'ranging')
        assert 'prophet_v2' in out['unmatched_models']

    def test_it_logs_an_error_not_a_shrug(self, engine, caplog):
        with caplog.at_level(logging.ERROR):
            engine.generate_weighted_ensemble({'prophet_v2': 0.05}, 'ranging')
        assert any('recognised NONE' in r.message for r in caplog.records)

    def test_genuine_agreement_on_zero_is_still_a_number(self, engine):
        # The other side of the distinction: models that DO participate and
        # agree on no move must return 0.0, not None.
        out = engine.generate_weighted_ensemble(
            {'LGBM_5m': 0.0, 'linear_60m': 0.0}, 'ranging')
        assert out['ensemble_prediction'] == 0.0


class TestPartialRecognitionIsReported:
    def test_recognised_models_still_produce_a_score(self, engine):
        out = engine.generate_weighted_ensemble(
            {'LGBM_5m': 0.02, 'prophet_v2': 99.0}, 'ranging')
        assert out['ensemble_prediction'] == pytest.approx(0.02)

    def test_the_ignored_model_is_named(self, engine, caplog):
        with caplog.at_level(logging.WARNING):
            out = engine.generate_weighted_ensemble(
                {'LGBM_5m': 0.02, 'prophet_v2': 99.0}, 'ranging')
        assert out['unmatched_models'] == ['prophet_v2']
        assert any('no known' in r.message for r in caplog.records)
