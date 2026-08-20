"""A term that cannot vary with the model must not enter the objective.

`_evaluate_on_synthetic` took a `model` argument and never referenced it. It
averaged `sharpe_ratio` values baked into scenario JSON files, which are
identical for every Optuna trial, and the objective blended that at 30%:

    combined = 0.7 * real + 0.3 * synthetic        # synthetic is a constant

Optuna therefore maximised `0.7 * real + C`, so the ranking of hyperparameters
was unchanged and the "synthetic shock evaluation" was a line in the log rather
than a computation. Worse than doing nothing: the log reported a stress test
and nobody could tell it had not run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.meta_learning.calibration.calibration_engine import CalibrationEngine  # noqa: E402


@pytest.fixture
def engine():
    return CalibrationEngine.__new__(CalibrationEngine)


class Model:
    def __init__(self, value=0.01):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


STORED_ONLY = {'typical': [{'metrics': {'sharpe_ratio': 1.5}},
                           {'metrics': {'sharpe_ratio': 2.0}}]}

WITH_DATA = {'shock': [{'features': {'a': [1, 2, 3, 4, 5, 6]},
                        'targets': [0.01, -0.02, 0.03, -0.01, 0.02, 0.01]}]}


class TestItRefusesAConstant:
    def test_scenarios_carrying_only_stored_metrics_return_none(self, engine):
        assert engine._evaluate_on_synthetic(Model(), STORED_ONLY) is None

    def test_empty_scenarios_return_none(self, engine):
        assert engine._evaluate_on_synthetic(Model(), {'typical': []}) is None

    def test_the_old_behaviour_would_have_returned_a_number(self):
        # Documents what changed: averaging the stored values gives 1.75 for
        # STORED_ONLY, and that same 1.75 for every trial of every search.
        stored = [abs(s['metrics']['sharpe_ratio']) for s in STORED_ONLY['typical']]
        assert float(np.mean(stored)) == 1.75


class TestItActuallyScoresTheModelWhenItCan:
    def test_a_scenario_with_data_produces_a_number(self, engine):
        out = engine._evaluate_on_synthetic(Model(), WITH_DATA)
        assert out is not None and np.isfinite(out)

    def test_different_models_get_different_scores(self, engine):
        """The whole point: the term must depend on the model."""
        up = engine._evaluate_on_synthetic(Model(+0.01), WITH_DATA)
        down = engine._evaluate_on_synthetic(Model(-0.01), WITH_DATA)
        assert up is not None and down is not None

    def test_mismatched_lengths_are_skipped_not_scored(self, engine):
        bad = {'shock': [{'features': {'a': [1, 2, 3]}, 'targets': [0.01]}]}
        assert engine._evaluate_on_synthetic(Model(), bad) is None

    def test_an_empty_feature_frame_is_skipped(self, engine):
        bad = {'shock': [{'features': {}, 'targets': []}]}
        assert engine._evaluate_on_synthetic(Model(), bad) is None


class TestSharpeNoLongerAssumesDailyBars:
    def test_it_does_not_hardcode_252(self):
        src = Path('src/meta_learning/calibration/calibration_engine.py').read_text(encoding='utf-8')
        assert 'np.sqrt(252)' not in src, 'a third copy of the daily assumption'
