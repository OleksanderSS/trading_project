"""The table both contextual subsystems needed and neither built.

ContextualModelSelector.select_models failed its first isinstance check
every time, because nothing ever put a fitted KnnSimilarityFinder or a
current_context Series into its `data` argument. The selector was
constructed on every run, logged at startup, and could not answer.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.meta_learning.memory.context_performance_history import (
    ContextPerformanceHistory,
)


class _Connection:
    def __init__(self, frame):
        self._frame = frame
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((query, list(params or [])))
        return self

    def fetchdf(self):
        return self._frame


class _DataManager:
    def __init__(self, frame):
        self.con = _Connection(frame)


def _history(rows):
    columns = ["context_fingerprint", "agent_id", "trades", "win_rate"]
    return ContextPerformanceHistory(
        _DataManager(pd.DataFrame(rows, columns=columns))
    )


def _spread(fingerprints, agents=("catboost", "lightgbm")):
    """One row per (fingerprint, agent), with a deterministic win rate."""
    return [
        {
            "context_fingerprint": fingerprint,
            "agent_id": agent,
            "trades": 5,
            "win_rate": 0.9 if agent == "catboost" else 0.2,
        }
        for fingerprint in fingerprints
        for agent in agents
    ]


FIVE = ["1|0|1", "1|1|1", "0|0|1", "-1|0|1", "1|0|-1"]


def test_resolved_outcomes_only():
    history = _history([])
    history.performance_by_context()

    query, params = history.data_manager.con.queries[0]
    assert "outcome IN (" in query
    assert "profitable" in params and "unprofitable" in params
    assert "pending" not in params and "neutral" not in params


def test_a_fitted_finder_and_current_context_come_back():
    history = _history(_spread(FIVE))

    inputs = history.similarity_inputs("1|0|1", n_neighbors=2)

    assert inputs is not None
    assert isinstance(inputs["current_context"], pd.Series)
    assert list(inputs["current_context"]) == [1.0, 0.0, 1.0]
    assert inputs["contexts_considered"] == 5
    outcomes = inputs["similarity_finder"].historical_outcomes
    assert set(outcomes.columns) == {"target_catboost", "target_lightgbm"}


def test_the_selector_can_actually_use_what_this_returns():
    """End to end, against the real ContextualModelSelector."""
    from src.analytics.context.contextual_model_selector import (
        ContextualModelSelector,
    )

    history = _history(_spread(FIVE))
    inputs = history.similarity_inputs("1|0|1", n_neighbors=3)

    chosen = ContextualModelSelector(["catboost", "lightgbm"]).select_models(
        "AAPL", "1|0|1", data=inputs
    )

    assert chosen == ["catboost"], "the model with the better contextual record"


def test_an_undecodable_fingerprint_returns_none():
    """A SHA-256 has no state vector; inventing one invents neighbours."""
    history = _history(_spread(FIVE))
    hashed = "67a9e31d5fb39bf5212a44f8ff79c9e423d088acc1c18f695f51c59f38091385"

    assert history.similarity_inputs(hashed) is None


def test_too_few_contexts_returns_none_rather_than_a_search_over_nothing():
    history = _history(_spread(["1|0|1", "0|0|1"]))

    assert history.similarity_inputs("1|0|1") is None


def test_an_empty_diary_returns_none():
    """Today's state: 19,305 rows, none resolved."""
    assert _history([]).similarity_inputs("1|0|1") is None


def test_fingerprints_of_a_different_width_are_dropped():
    """fingerprint_to_vec silently drops unparseable tokens, so a junk-bearing
    fingerprint yields a shorter vector -- comparing it would line driver 12
    up against driver 13."""
    rows = _spread([*FIVE, "1|junk|0|1|1"])
    history = _history(rows)

    inputs = history.similarity_inputs("1|0|1")

    assert inputs["contexts_considered"] == 5


def test_a_model_absent_from_a_context_is_not_scored_as_a_zero_win_rate():
    """NaN, not 0.0 -- otherwise a model that never traded there ranks below
    one that traded and sometimes lost."""
    rows = _spread(FIVE)
    rows = [row for row in rows if not (
        row["context_fingerprint"] == "1|1|1" and row["agent_id"] == "lightgbm"
    )]

    inputs = _history(rows).similarity_inputs("1|0|1")
    outcomes = inputs["similarity_finder"].historical_outcomes

    assert pd.isna(outcomes.loc["1|1|1", "target_lightgbm"])


def test_outcomes_stay_row_aligned_with_the_fitted_features():
    """The selector reads historical_outcomes.iloc[neighbour_positions], so a
    reordering would attribute one context's record to another."""
    inputs = _history(_spread(FIVE)).similarity_inputs("1|0|1")
    finder = inputs["similarity_finder"]

    positions, _ = finder.find_similar_situations(inputs["current_context"])
    assert len(positions) > 0
    assert finder.historical_outcomes.iloc[positions].notna().any().any()


def test_a_database_failure_degrades_to_empty_rather_than_raising():
    class _Broken:
        con = None

    history = ContextPerformanceHistory(_Broken())

    assert history.performance_by_context().empty
    assert history.similarity_inputs("1|0|1") is None


@pytest.mark.parametrize("fingerprint", ["", "normal", "unknown"])
def test_placeholder_fingerprints_return_none(fingerprint):
    assert _history(_spread(FIVE)).similarity_inputs(fingerprint) is None
