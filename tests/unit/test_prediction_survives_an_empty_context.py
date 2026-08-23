"""One context with no rows stopped stages 5, 6 and 7 entirely.

2026-08-23 was the first run in this project's history in which the final
stages executed at all. Stage 5 died three minutes in:

    IndexError: index -1 is out of bounds for axis 0 with size 0
    prediction_generator.py:168  pred_value = raw_prediction[-1] if ...

A context whose slice had no bars produced an empty prediction array, and the
code took its last element. Two things made that fatal rather than skippable:

  the guard      `isinstance(raw_prediction, np.ndarray)` is True for an empty
                 array, and so is `hasattr(p, '__len__')` in anomaly_engine.
                 Both checks pass and neither means there is an element.
  the except     the loop over contexts caught (ValueError, TypeError,
                 KeyError, AttributeError). IndexError was not in the tuple,
                 so one context ended the run -- the same narrow-tuple shape
                 the silent-failure contract already counts 653 of.

A per-context failure has to stay per-context. Stage 5 has 83 of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.prediction.prediction_generator import PredictionGenerator


@pytest.mark.parametrize("value,expected", [
    (np.array([]), None),
    (np.array([1.0, 2.0, 3.0]), 3.0),
    (np.array([[1.0], [2.0]]), None),      # 2-D: last row, not a scalar
    ([], None),
    ([1.0, 5.0], 5.0),
    (pd.Series([], dtype=float), None),
    (0.5, 0.5),
])
def test_the_last_value_of_nothing_is_nothing(value, expected):
    got = PredictionGenerator._last_value(value)
    if expected is None:
        assert got is None or (hasattr(got, "__len__") and len(got) == 1)
    else:
        assert got == expected


def test_an_empty_input_frame_yields_no_prediction_rather_than_raising():
    """The exact case: a context with no bars to predict on."""
    generator = PredictionGenerator.__new__(PredictionGenerator)
    import logging
    generator.logger = logging.getLogger("probe")

    class _Model:
        def predict(self, X):
            return np.array([])

    empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
    raw, weights = generator.generate_single_model_prediction(
        {"linear": _Model()}, "linear", empty, ["a"]
    )
    assert raw is None and weights == {}


def test_a_model_returning_an_empty_array_is_survived():
    """Rows went in, nothing came out. That is the model's answer, not a crash."""
    generator = PredictionGenerator.__new__(PredictionGenerator)
    import logging
    generator.logger = logging.getLogger("probe")

    class _Model:
        def predict(self, X):
            return np.array([])

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    raw, weights = generator.generate_single_model_prediction(
        {"linear": _Model()}, "linear", frame, ["a"]
    )
    assert raw is None and weights == {}


def test_one_failing_context_does_not_end_the_stage():
    """The narrow except tuple decided which failures were fatal, by accident."""
    import ast
    import inspect

    from src.pipeline.stages.prediction import orchestrator as module

    source = inspect.getsource(module._generate_predictions_for_contexts) \
        if hasattr(module, "_generate_predictions_for_contexts") \
        else inspect.getsource(module.PredictionStage._generate_predictions_for_contexts)

    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    handlers = [
        node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
    ]
    assert handlers, "the per-context loop no longer handles failures at all"
    for handler in handlers:
        if handler.type is None:
            continue
        name = ast.unparse(handler.type)
        assert name == "Exception", (
            f"the per-context loop catches {name}; anything outside that list "
            "ends stages 5, 6 and 7 for every remaining context"
        )
