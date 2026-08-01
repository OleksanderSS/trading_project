"""Contextual model weights decide how much each model's forecast counts.

Used live by StackedEnsemble (stacked_ensemble.py:130) and ConsensusEngine
(consensus_engine.py:182), so an error here moves predictions directly.

The performance score was computed as

    AVG(CASE WHEN decision_type = 'training' THEN COALESCE(model_prediction, 0)
             WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END)

which averages two incompatible quantities: a 0/1 win rate, and -- for
training rows -- `model_prediction`, a field log_training_event fills with
float(metrics), i.e. a TRAINING METRIC, not a prediction. Those metrics are
unbounded and can be very negative.

Measured on the live diary (19,305 rows, all of them training):

    linear     performance_score  -13820.5   (its metrics reach -1,420,512)
    catboost                        0.1838
    xgboost                        -0.0082

and for the busiest fingerprint the weights came out as

    linear -0.25345, svm 0.26211, random_forest 0.24370, ...   sum = 1.0

A NEGATIVE ensemble weight does not mean "ignore this model", it means
SUBTRACT its forecast -- the model's signal gets inverted. And because
normalisation only checks the total, the weights still summed to 1.0 and
looked healthy.

Weights now come from realized outcomes only. Training rows say nothing
about how a model performs in a context; they carry outcome='neutral' and no
P&L. With no realized outcomes yet, the query returns nothing and the caller
falls back to equal weights, which is the honest state of the evidence.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.meta_learning.memory.contextual_weight_calculator import (
    ContextualWeightCalculator,
)


@pytest.fixture()
def calculator():
    instance = ContextualWeightCalculator.__new__(ContextualWeightCalculator)
    instance.logger = logging.getLogger("contextual-weights-test")
    return instance


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["agent_id", "performance_score", "avg_pnl"]
    )


def test_a_negative_score_never_becomes_a_negative_weight(calculator):
    """The live failure: linear scored -13,820 and drew a weight of -0.25."""
    weights = calculator._calculate_weights_from_dataframe(
        _frame([("linear", -13820.5, 0.0), ("catboost", 0.18, 0.0)])
    )

    assert all(weight >= 0 for weight in weights.values())
    assert weights["linear"] == 0.0
    assert weights["catboost"] == pytest.approx(1.0)


def test_weights_sum_to_one(calculator):
    weights = calculator._calculate_weights_from_dataframe(
        _frame([("a", 0.6, 0.0), ("b", 0.3, 0.0), ("c", 0.1, 0.0)])
    )

    assert sum(weights.values()) == pytest.approx(1.0)


def test_a_better_model_gets_more_weight(calculator):
    weights = calculator._calculate_weights_from_dataframe(
        _frame([("good", 0.9, 0.0), ("poor", 0.1, 0.0)])
    )

    assert weights["good"] > weights["poor"]


def test_all_zero_scores_mean_no_evidence_not_zero_weights(calculator):
    """Returning zeros would hand the ensemble a set of useless weights; {}
    is the documented signal for equal weighting."""
    assert calculator._calculate_weights_from_dataframe(
        _frame([("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    ) == {}


def test_an_empty_frame_yields_no_weights(calculator):
    assert calculator._calculate_weights_from_dataframe(
        pd.DataFrame(columns=["agent_id", "performance_score", "avg_pnl"])
    ) == {}


def test_profit_raises_a_models_share(calculator):
    """avg_pnl is the tie-breaker between models with equal win rates."""
    weights = calculator._calculate_weights_from_dataframe(
        _frame([("profitable", 0.5, 1.0), ("flat", 0.5, 0.0)])
    )

    assert weights["profitable"] > weights["flat"]


@pytest.mark.parametrize("method", [
    "get_contextual_model_weights",
    "get_contextual_model_weights_by_pattern_seq",
])
def test_both_queries_count_only_realized_outcomes(method):
    """Training rows carry outcome='neutral' and no P&L, and their
    model_prediction column holds a training metric."""
    import inspect

    source = inspect.getsource(getattr(ContextualWeightCalculator, method))

    assert "outcome IN ('profitable', 'unprofitable', 'break_even')" in source
    assert "decision_type = 'training'" not in source


def test_the_live_diary_produces_no_weights_yet():
    """Every row in the diary today is a training row, so there is no
    realized evidence -- and the honest answer is equal weights, not the
    -0.25 the old query invented."""
    import duckdb

    connection = duckdb.connect("data/trading_data.duckdb", read_only=True)
    realized = connection.execute(
        "SELECT COUNT(*) FROM experience_diary "
        "WHERE outcome IN ('profitable','unprofitable','break_even')"
    ).fetchone()[0]
    connection.close()

    if realized:
        pytest.skip(
            "paper trading has produced realized outcomes; weights are now "
            "meaningful and this expectation no longer holds"
        )
    assert realized == 0
