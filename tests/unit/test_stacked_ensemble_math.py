"""Feature importances are not signed weights, and abs-normalisation is shrinkage.

`StackedEnsemble` combined base-model predictions by dot-producting them with
the meta-model's `feature_importances_`. A tree's importance is a strictly
POSITIVE measure of information gain: a base model that perfectly
ANTI-correlates with the target earns a high importance -- it splits
beautifully -- and its prediction was then added with a plus. The sign is
simply not present in the quantity.

It also divided the weights by the sum of their ABSOLUTE values. Coefficients
[1.5, -0.5] carry a net weight of 1.0; dividing by 2.0 halves the magnitude of
every prediction. KellyCriterion sizes on that magnitude, so positions came out
arbitrarily small -- smaller the more the meta-model disagreed with itself.
And `intercept_` was dropped, which for a return forecast is a bias.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ensembling.stacked_ensemble import StackedEnsemble  # noqa: E402


class LinearMeta:
    def __init__(self, coef, intercept=0.0):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = np.asarray([intercept], dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_[0]


class TreeMeta:
    """Importances say which column mattered; predict() says what it thinks."""

    def __init__(self, importances, output):
        self.feature_importances_ = np.asarray(importances, dtype=float)
        self._output = output

    def predict(self, X):
        return np.full(len(X), self._output, dtype=float)


@pytest.fixture
def ensemble():
    e = StackedEnsemble.__new__(StackedEnsemble)
    e.feature_names = ['m1', 'm2']
    e.is_trained = True

    class _Diary:
        def get_contextual_model_weights(self, fp):
            return {}
    e.diary_engine = _Diary()

    class _Router:
        def adjust_weights(self, preds, params):
            return {}
    e.dynamic_router = _Router()
    return e


def frame(a, b):
    return pd.DataFrame({'m1': a, 'm2': b})


class TestTreeMetaModelsAreAsked_NotParsed:
    def test_a_tree_stacker_returns_its_own_prediction(self, ensemble):
        ensemble.meta_model = TreeMeta([0.9, 0.1], output=0.042)
        out = ensemble._predict_stacked(frame([1.0, 1.0], [-1.0, -1.0]))
        assert np.allclose(out.final_signal, 0.042), (
            'a tree prediction is not a linear combination of its inputs')

    def test_an_anticorrelated_model_is_not_added_positively(self, ensemble):
        """The defect in one test.

        m2 is the exact negative of the target. A tree gives it a HIGH
        importance because it splits perfectly, and the old dot product then
        added it with a plus.
        """
        ensemble.meta_model = TreeMeta([0.05, 0.95], output=-1.0)
        out = ensemble._predict_stacked(frame([0.0, 0.0], [1.0, 1.0]))
        old_style = np.dot(np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([0.05, 0.95]))
        assert not np.allclose(out.final_signal, old_style)
        assert np.allclose(out.final_signal, -1.0)


class TestLinearMetaModelsKeepTheirMagnitude:
    def test_the_intercept_is_not_dropped(self, ensemble):
        ensemble.meta_model = LinearMeta([1.0, 0.0], intercept=0.01)
        out = ensemble._predict_stacked(frame([0.0, 0.0], [0.0, 0.0]))
        assert np.allclose(out.final_signal, 0.01), 'the intercept shifts every forecast'

    def test_opposing_coefficients_do_not_shrink_the_signal(self, ensemble):
        """[1.5, -0.5] is a net weight of 1.0, not 0.5."""
        ensemble.meta_model = LinearMeta([1.5, -0.5])
        out = ensemble._predict_stacked(frame([1.0], [0.0]))
        abs_normalised = 1.5 / 2.0        # what the old code produced
        assert not np.isclose(out.final_signal[0], abs_normalised)
        assert np.isclose(out.final_signal[0], 1.5)

    def test_weights_that_cancel_are_left_alone_not_amplified(self, ensemble):
        # Signed sum near zero means the models genuinely disagree. Rescaling
        # by it would manufacture a signal out of disagreement.
        ensemble.meta_model = LinearMeta([1.0, -1.0])
        out = ensemble._predict_stacked(frame([1.0], [0.0]))
        assert np.isfinite(out.final_signal[0])
        assert abs(out.final_signal[0]) <= 1.0


class TestItDoesNotFallOver:
    def test_a_meta_model_whose_predict_raises_falls_back_to_the_average(self, ensemble):
        class Broken(TreeMeta):
            def predict(self, X):
                raise RuntimeError('no')
        ensemble.meta_model = Broken([0.5, 0.5], output=0.0)
        out = ensemble._predict_stacked(frame([1.0], [3.0]))
        assert np.allclose(out.final_signal, 2.0), 'the mean is at least defined'
