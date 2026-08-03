"""Champion selection compared R2 against classification accuracy.

_build_model_groups inferred per model whether it was a regression from
whether its metrics dict held r2/mse, stored either R2 or accuracy under one
key named 'accuracy', and _get_champion_model_for_target took max() over it.

R2 ranges (-inf, 1] and sits near zero on financial returns. Accuracy ranges
[0, 1] and a coin flip scores 0.50. Ranking one against the other puts a
useless classifier above a genuinely useful regressor every time -- and a
single model reporting a different metric set from its peers on the SAME
target was enough to trigger it.

The metric is now chosen by the target's declared type, so every candidate
for a target is read on one scale.
"""
from __future__ import annotations

import logging

import pytest

from src.pipeline.stages.trading.recommendation_engine import (
    TradingRecommendationEngine,
)


@pytest.fixture()
def engine():
    instance = object.__new__(TradingRecommendationEngine)
    instance.logger = logging.getLogger("recommendation-ranking-test")
    instance._target_types_cache = {
        "target_return_1d": "regression",
        "target_up_1d": "classification_binary",
        "target_multi_1d": "classification_multiclass",
    }
    return instance


def test_a_regression_target_is_ranked_on_r2(engine):
    assert engine._ranking_score("target_return_1d", {"r2": 0.05, "accuracy": 0.9}) == (
        0.05, "r2",
    )


def test_a_classification_target_is_ranked_on_accuracy(engine):
    assert engine._ranking_score("target_up_1d", {"r2": 0.9, "accuracy": 0.62}) == (
        0.62, "accuracy",
    )


def test_a_useless_classifier_no_longer_outranks_a_useful_regressor(engine):
    """The defect, as it would actually play out on one target."""
    heavy = {"AAPL_target_return_1d": [
        {"model_type": "lstm", "accuracy": engine._ranking_score(
            "target_return_1d", {"accuracy": 0.52})[0], "score_metric": "score"},
    ]}
    light = {"AAPL_target_return_1d": [
        {"model_type": "catboost", "accuracy": engine._ranking_score(
            "target_return_1d", {"r2": 0.05})[0], "score_metric": "r2"},
    ]}

    # The classifier's 0.52 is not on the R2 scale at all; it now scores -inf
    # because it reported no r2 and no score.
    champion = engine._get_champion_model_for_target(
        "AAPL_target_return_1d", heavy, light
    )

    assert champion["model_type"] == "catboost"


def test_a_missing_metric_scores_worse_than_a_bad_one(engine):
    """0.0 is a real R2 (no better than the mean) and a real accuracy (never
    right), so defaulting to it lets an unmeasured model outrank a measured
    failure."""
    unmeasured, _ = engine._ranking_score("target_return_1d", {})
    measured_badly, _ = engine._ranking_score("target_return_1d", {"r2": -0.4})

    assert unmeasured < measured_badly


def test_score_is_used_only_after_the_type_metric_is_missing(engine):
    assert engine._ranking_score(
        "target_return_1d", {"score": 0.3}
    ) == (0.3, "score")
    assert engine._ranking_score(
        "target_return_1d", {"r2": 0.1, "score": 0.9}
    ) == (0.1, "r2")


def test_a_mixed_ranking_is_announced(engine, caplog):
    groups = {"AAPL_target_return_1d": [
        {"model_type": "catboost", "accuracy": 0.05, "score_metric": "r2"},
        {"model_type": "linear", "accuracy": 0.30, "score_metric": "score"},
    ]}

    with caplog.at_level(logging.WARNING):
        engine._get_champion_model_for_target("AAPL_target_return_1d", groups, {})

    assert any("different metrics" in record.message for record in caplog.records)


def test_no_usable_metric_means_no_champion(engine, caplog):
    """Returning the least-bad of several unmeasured models would present a
    guess as a selection."""
    groups = {"AAPL_target_return_1d": [
        {"model_type": "svm", "accuracy": float("-inf"), "score_metric": "missing"},
    ]}

    with caplog.at_level(logging.WARNING):
        champion = engine._get_champion_model_for_target(
            "AAPL_target_return_1d", groups, {}
        )

    assert champion is None
    assert any("no champion" in record.message.lower() for record in caplog.records)


def test_an_empty_group_still_returns_none(engine):
    assert engine._get_champion_model_for_target("AAPL_x", {}, {}) is None


def test_a_non_dict_metrics_payload_does_not_raise(engine):
    assert engine._ranking_score("target_return_1d", None)[1] == "missing"
