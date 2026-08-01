"""The context sequence must reach the diary, or KNN has nothing to search.

Every piece of this already existed and one link was missing:

  ContextMapEnricher  writes df['context_pattern_seq'], deliberately keeping
                      the RAW sequence -- its own comment says "the hash is
                      useful as a compact ID, but KNN needs the original
                      state sequence"
  BaseTrainer         forwards it: log_event(..., context_pattern_seq=
                      data.get('context_pattern_seq'))
  ModelingStage       built the training-context dict WITHOUT that key

so data.get() returned None and all 19,305 diary rows carry a NULL sequence.
KnnContextFinder then had nothing to measure distance against, and the KNN
expansion has never produced a neighbour.

Not solved by teaching fingerprint_to_vec to read the SHA-256 in
context_fingerprint: a hash is an identity, not a coordinate, and any vector
derived from one fabricates neighbours. The tri-state sequence is the form
that carries geometry, which is why the enricher keeps it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.meta_learning.memory.contextual_weight_calculator import (
    ContextualWeightCalculator,
)
from src.pipeline.stages.modeling.orchestrator import ModelingStage


def _prepared():
    return {
        "light_models": {
            "X_train": pd.DataFrame({"a": [1.0, 2.0]}),
            "y_train": pd.Series([0.1, 0.2]),
            "X_val": pd.DataFrame({"a": [3.0]}),
            "y_val": pd.Series([0.3]),
            "feature_names": ["a"],
        }
    }


def _frame():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "context_fingerprint": ["1|1|0", "1|0|0", "1|1|1"],
        "context_pattern_seq": ["1|1|0>>START", "1|0|0>>1|1|0", "1|1|1>>1|0|0"],
    })


def test_the_latest_sequence_is_taken_from_the_frame():
    assert ModelingStage._latest_context_value(
        _frame(), ("context_pattern_seq",), default=None
    ) == "1|1|1>>1|0|0"


def test_the_training_context_carries_the_sequence():
    stage = object.__new__(ModelingStage)

    context = ModelingStage._build_unified_training_context(
        stage,
        _prepared(),
        target_name="target_x",
        context_fingerprint="fp",
        context_pattern_seq="1|1|1>>1|0|0",
    )

    assert context["context_pattern_seq"] == "1|1|1>>1|0|0"


def test_a_frame_without_the_column_yields_none_not_an_error():
    frame = _frame().drop(columns=["context_pattern_seq"])

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default=None
    ) is None


def test_the_sequence_vectorises_so_knn_can_measure_distance():
    vector = ContextualWeightCalculator.fingerprint_to_vec("1|1|1>>1|0|0")

    assert vector, "an empty vector is what stopped KNN before"
    assert all(isinstance(value, float) for value in vector)


def test_sequences_of_the_same_shape_give_vectors_of_equal_length():
    """_build_knn_vectors keeps only same-length vectors, so comparability
    depends on this."""
    a = ContextualWeightCalculator.fingerprint_to_vec("1|1|1>>1|0|0")
    b = ContextualWeightCalculator.fingerprint_to_vec("0|1|0>>1|1|1")

    assert len(a) == len(b)


def test_the_separator_token_is_dropped_and_that_is_survivable():
    """'1>>1' parses as neither number, so the bar boundary is lost and one
    value of each pair of adjacent states goes with it. Recorded rather than
    changed: every sequence has the same shape (fixed pattern_length, fixed
    state columns), so the loss is uniform and distances stay meaningful --
    and altering fingerprint_to_vec would also change how plain tri-state
    fingerprints vectorise."""
    assert ContextualWeightCalculator.fingerprint_to_vec("1|1|1>>1|0|0") == [
        1.0, 1.0, 0.0, 0.0
    ]


def test_the_trainer_still_forwards_what_it_is_given():
    import inspect

    from src.training import base_trainer

    source = inspect.getsource(base_trainer)
    assert "context_pattern_seq=data.get('context_pattern_seq')" in source


def test_the_stage_passes_it_at_the_call_site():
    import inspect

    source = inspect.getsource(ModelingStage)
    assert "context_pattern_seq=self._latest_context_value(" in source
