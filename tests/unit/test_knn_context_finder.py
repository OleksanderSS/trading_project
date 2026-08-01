"""KNN expansion for contextual weights: what it can and cannot do here.

Measured against the live diary (19,305 rows) rather than assumed:

- `context_fingerprint` holds either a SHA-256 of a JSON payload
  (ModelingStage._build_context_fingerprint: ticker, timeframe, target,
  pattern id, feature names, LAST FEATURE VALUES) or the literal 'normal'.
  Both vectorise to [], so _build_knn_vectors bails and the KNN expansion has
  never run. A hash is an identity, not a coordinate -- there is no
  similarity to compute between two of them, and manufacturing a vector from
  one would fabricate neighbours.

- `context_pattern_seq`, the tri-state form ('1|1>>1|0') that CAN be
  vectorised and is what similarity was designed around, is NULL in all
  19,305 rows.

So the machinery is sound and starved. This test pins the honest behaviour:
skip, say so once, and fall back to exact matches -- never invent a
neighbour.
"""
from __future__ import annotations

import logging

import pytest

from src.meta_learning.memory.contextual_weight_calculator import (
    ContextualWeightCalculator,
)
from src.meta_learning.memory.knn_context_finder import KnnContextFinder


@pytest.fixture()
def finder():
    instance = KnnContextFinder.__new__(KnnContextFinder)
    instance.logger = logging.getLogger("knn-context-test")
    instance.weight_calculator = None
    instance.data_manager = None
    KnnContextFinder._reported_unvectorisable = False
    return instance


HASH = "67a9e31d5fb39bf5212a44f8ff79c9e423d088acc1c18f695f51c59f38091385"


@pytest.mark.parametrize("fingerprint", [HASH, "normal", "elevated", ""])
def test_the_live_fingerprint_forms_carry_no_vector(fingerprint):
    assert ContextualWeightCalculator.fingerprint_to_vec(fingerprint) == []


def test_the_tristate_form_does_vectorise():
    assert ContextualWeightCalculator.fingerprint_to_vec("1|1>>1|0") == [1.0, 0.0]


def test_an_unvectorisable_fingerprint_skips_knn_rather_than_guessing(finder):
    assert finder._build_knn_vectors(HASH, [HASH, HASH, HASH], 1) is None


def test_the_skip_is_reported_once_not_on_every_call(finder, caplog):
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            finder._build_knn_vectors(HASH, [HASH], 1)

    warnings = [r for r in caplog.records if "KNN contextual weights" in r.getMessage()]
    assert len(warnings) == 1, "one explanation, not one per call"


def test_the_warning_does_not_print_the_whole_fingerprint(finder, caplog):
    with caplog.at_level(logging.WARNING):
        finder._build_knn_vectors(HASH, [HASH], 1)

    message = next(r.getMessage() for r in caplog.records if "KNN" in r.getMessage())
    assert HASH not in message


def test_neighbour_weights_are_normalised(finder):
    class _Calculator:
        def get_contextual_model_weights(self, fingerprint):
            return {"a": 0.6, "b": 0.4} if fingerprint == "x" else {"a": 0.2, "b": 0.8}

    finder.weight_calculator = _Calculator()
    weights = finder._aggregate_neighbor_weights(["x", "y"])

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["b"] > weights["a"]


def test_neighbours_with_no_weights_yield_no_weights(finder):
    class _Empty:
        def get_contextual_model_weights(self, fingerprint):
            return {}

    finder.weight_calculator = _Empty()
    assert finder._aggregate_neighbor_weights(["x", "y"]) == {}


def test_all_zero_neighbour_weights_mean_no_evidence(finder):
    """Returning the raw zeros would hand the ensemble meaningless weights."""
    class _Zeros:
        def get_contextual_model_weights(self, fingerprint):
            return {"a": 0.0, "b": 0.0}

    finder.weight_calculator = _Zeros()
    assert finder._aggregate_neighbor_weights(["x"]) == {}


def test_the_live_diary_has_no_pattern_sequences_to_search():
    """If this ever fails, similarity has become possible and the KNN path
    is worth exercising for real."""
    import duckdb

    connection = duckdb.connect("data/trading_data.duckdb", read_only=True)
    filled = connection.execute(
        "SELECT COUNT(context_pattern_seq) FROM experience_diary"
    ).fetchone()[0]
    connection.close()

    if filled:
        pytest.skip(
            f"context_pattern_seq is now populated ({filled} rows); KNN "
            "similarity can run and should be tested against real neighbours"
        )
    assert filled == 0
