"""Permutation importance is the increase in ERROR, and error needs y.

The fallback path (models with neither feature_importances_ nor coef_)
computed

    base_error = np.mean(np.abs(base_pred))   # "Mean Absolute Error"

with no true values anywhere in it. That measures how much the average
prediction MAGNITUDE moves when a feature is shuffled -- which a decisive
feature can leave untouched, and which a feature that merely shifts the
output level can dominate.

Demonstrated on a model that is literally `3 * signal`, with a second pure
noise column. With y_true the ranking is signal 1.0, noise 0.0.

Without it, shuffling `signal` destroys accuracy while barely moving the
mean absolute prediction -- the same numbers in a different order. Both raw
scores come out near zero and near equal, and normalisation then promotes
whichever rounding error happened to be larger to 1.0. So the old statistic
did not merely understate the useful feature: it could not separate the two
at all, and produced a confident-looking ranking out of noise.

Not a live defect: ExplainabilityCalculator is instantiated by
technical_analysis_enricher (line 58) and none of its methods are ever
called -- the same dormant-construction pattern as four of the five
FeatureGuards.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.explainability_calculator import (
    ExplainabilityCalculator,
)


class _BlackBox:
    """No feature_importances_, no coef_ -- forces the permutation path."""

    def predict(self, frame):
        return 3.0 * frame["signal"].to_numpy()


class _Trees:
    feature_importances_ = np.array([0.7, 0.3])

    def predict(self, frame):  # pragma: no cover - not used by this path
        return np.zeros(len(frame))


class _Linear:
    coef_ = np.array([-2.0, 0.5])

    def predict(self, frame):  # pragma: no cover - not used by this path
        return np.zeros(len(frame))


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "signal": rng.normal(size=400),
        "noise": rng.normal(size=400),
    })
    target = 3.0 * frame["signal"] + rng.normal(0, 0.1, len(frame))
    return frame, target


def test_the_informative_feature_wins_when_truth_is_supplied(data):
    frame, target = data

    importance = ExplainabilityCalculator.analyze_feature_importance(
        _BlackBox(), frame, ["signal", "noise"], y_true=target
    )

    assert importance["signal"] > importance["noise"]


def test_pure_noise_scores_zero(data):
    frame, target = data

    importance = ExplainabilityCalculator.analyze_feature_importance(
        _BlackBox(), frame, ["signal", "noise"], y_true=target
    )

    assert importance["noise"] == pytest.approx(0.0, abs=1e-9)


def test_a_shuffle_that_helps_counts_as_no_importance(data):
    """Negative "increase in error" means the feature carried no signal --
    importance zero, not importance |difference|."""
    frame, target = data

    importance = ExplainabilityCalculator.analyze_feature_importance(
        _BlackBox(), frame, ["signal", "noise"], y_true=target
    )

    assert all(value >= 0 for value in importance.values())


def test_without_truth_the_degraded_mode_is_announced(data, caplog):
    frame, _ = data

    with caplog.at_level(logging.WARNING):
        ExplainabilityCalculator.analyze_feature_importance(
            _BlackBox(), frame, ["signal", "noise"]
        )

    assert any("NOT an error metric" in r.getMessage() for r in caplog.records)


def test_the_old_statistic_cannot_separate_the_features(data):
    """Why the change was needed, stated precisely.

    Shuffling `signal` destroys accuracy but barely moves the MEAN ABSOLUTE
    PREDICTION -- it reorders the same numbers. So the old metric produced
    two near-zero, near-equal raw scores, and whichever noise happened to be
    larger became 1.0 after normalisation. Not "zero for the useful
    feature", as first reported: an arbitrary ranking driven by rounding."""
    frame, target = data
    model = _BlackBox()

    base = float(np.mean(np.abs(model.predict(frame))))
    shuffled = frame.copy()
    shuffled["signal"] = np.random.default_rng(1).permutation(shuffled["signal"])
    permuted = float(np.mean(np.abs(model.predict(shuffled))))

    # The old statistic barely notices, in relative terms...
    assert abs(permuted - base) / base < 0.10

    # ...while the real error metric more than triples.
    base_error = float(np.mean(np.abs(target - model.predict(frame))))
    permuted_error = float(np.mean(np.abs(target - model.predict(shuffled))))
    assert permuted_error > base_error * 3


def test_tree_models_use_their_native_importances(data):
    frame, target = data

    importance = ExplainabilityCalculator.analyze_feature_importance(
        _Trees(), frame, ["signal", "noise"], y_true=target
    )

    assert importance["signal"] == pytest.approx(0.7)
    assert importance["noise"] == pytest.approx(0.3)


def test_linear_models_use_absolute_coefficients(data):
    frame, target = data

    importance = ExplainabilityCalculator.analyze_feature_importance(
        _Linear(), frame, ["signal", "noise"], y_true=target
    )

    assert importance["signal"] > importance["noise"], "sign must not decide rank"
    assert sum(importance.values()) == pytest.approx(1.0)


def test_results_are_normalised_and_sorted(data):
    frame, target = data

    importance = ExplainabilityCalculator.analyze_feature_importance(
        _Trees(), frame, ["signal", "noise"], y_true=target
    )
    values = list(importance.values())

    assert sum(values) == pytest.approx(1.0)
    assert values == sorted(values, reverse=True)


def test_a_single_row_explanation_still_works(data):
    frame, _ = data

    result = ExplainabilityCalculator.explain_single_prediction(
        _Trees(), frame.head(1)
    )

    assert set(result) == {"signal", "noise"}


def test_a_multi_row_frame_is_refused_for_single_prediction(data):
    frame, _ = data

    assert ExplainabilityCalculator.explain_single_prediction(_Trees(), frame) == {}
