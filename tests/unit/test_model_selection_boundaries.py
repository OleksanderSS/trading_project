"""Which models a ticker trains, and the guards around that choice.

_select_models_for_ticker has four ways to answer and only one of them fires
in the shipped configuration. These tests pin down which, and pin the guards
on the branches that do not fire yet, because "unreachable today" is exactly
how a latent bug survives to the day it becomes reachable.
"""
from __future__ import annotations

import pytest

from src.analytics.context.contextual_model_selector import ContextualModelSelector
from src.training.unified_training_manager import UnifiedTrainingManager


LIGHT = ["catboost", "lightgbm", "xgboost", "random_forest", "linear", "svm", "knn"]
HEAVY = ["mlp", "cnn", "lstm", "gru", "transformer", "tabnet", "autoencoder"]


class _NoHistory:
    """Stands in for ContextPerformanceHistory without opening a database.

    Returning None is also the honest answer today: the diary holds no
    resolved outcomes, so a real history has nothing to fit either.
    """

    def similarity_inputs(self, fingerprint, **kwargs):
        return None


class _Manager:
    """Exercises the selection logic without building trainers or an arena."""

    def __init__(self, models_config, recommendation=None, history=None):
        self._models_config = models_config
        self.logger = __import__("logging").getLogger("test")
        self.context_selector = _StubSelector(recommendation)
        self._context_history = history or _NoHistory()

    class _ConfigManager:
        def __init__(self, models_config):
            self._models_config = models_config

        def get_config(self, key, default=None):
            return self._models_config if key == "models" else default

    @property
    def config_manager(self):
        return self._ConfigManager(self._models_config)

    _permitted_model_names = staticmethod(
        UnifiedTrainingManager._permitted_model_names
    )
    _select_models_for_ticker = UnifiedTrainingManager._select_models_for_ticker
    _get_available_model_names = UnifiedTrainingManager._get_available_model_names
    # The real merge logic, not a stub: it decides what select_models sees.
    _with_similarity_inputs = UnifiedTrainingManager._with_similarity_inputs
    _performance_history = UnifiedTrainingManager._performance_history


class _StubSelector:
    def __init__(self, recommendation):
        self._recommendation = recommendation

    def select_models(self, ticker, context_fingerprint, data=None):
        return self._recommendation


def test_the_light_category_decides_in_the_shipped_configuration():
    """models.yaml sets categories.light, so nothing below it is consulted."""
    manager = _Manager(
        {"categories": {"light": LIGHT, "heavy": HEAVY}},
        recommendation=["lstm"],
    )

    assert manager._select_models_for_ticker("AAPL", {}) == LIGHT


def test_a_heavy_recommendation_cannot_leak_onto_the_local_path():
    """The hybrid split says heavy models are Colab's job.

    select_models ranks purely on historical performance and knows nothing
    about light vs heavy, so on the day it starts firing it could hand back
    'lstm' and the local pipeline would train a neural net on the laptop.
    """
    manager = _Manager(
        {"enabled_types": LIGHT},
        recommendation=["lstm"],
    )

    assert manager._select_models_for_ticker("AAPL", {}) == LIGHT


def test_a_permitted_recommendation_is_honoured():
    manager = _Manager(
        {"enabled_types": LIGHT},
        recommendation=["catboost"],
    )

    assert manager._select_models_for_ticker("AAPL", {}) == ["catboost"]


def test_a_mixed_recommendation_keeps_only_the_permitted_part():
    manager = _Manager(
        {"enabled_types": LIGHT},
        recommendation=["lstm", "catboost"],
    )

    assert manager._select_models_for_ticker("AAPL", {}) == ["catboost"]


def test_no_configured_opinion_means_no_filtering():
    """An empty permitted set must not be read as 'allow nothing'."""
    manager = _Manager({}, recommendation=["lstm"])

    assert manager._select_models_for_ticker("AAPL", {}) == ["lstm"]


@pytest.mark.parametrize(
    "models_config, expected",
    [
        ({"enabled_types": LIGHT}, set(LIGHT)),
        ({"categories": {"light": LIGHT, "heavy": HEAVY}}, set()),
        ({}, set()),
    ],
)
def test_permitted_names_ignores_categories(models_config, expected):
    """Unioning light+heavy would readmit the models the split excluded."""
    assert UnifiedTrainingManager._permitted_model_names(models_config) == expected


def test_the_fitted_finder_is_handed_to_the_selector():
    """select_models failed its first isinstance check every time, because
    nothing put current_context or similarity_finder into `data`."""
    import pandas as pd

    seen = {}

    class _Recording(_StubSelector):
        def select_models(self, ticker, context_fingerprint, data=None):
            seen.update(data or {})
            return ["catboost"]

    class _History:
        def similarity_inputs(self, fingerprint, **kwargs):
            return {
                "current_context": pd.Series([1.0, 0.0]),
                "similarity_finder": object(),
                "contexts_considered": 12,
            }

    manager = _Manager({"enabled_types": LIGHT}, history=_History())
    manager.context_selector = _Recording(None)

    manager._select_models_for_ticker("AAPL", {"context_fingerprint": "1|0"})

    assert "current_context" in seen
    assert "similarity_finder" in seen


def test_a_caller_that_already_supplied_a_finder_is_not_overridden():
    import pandas as pd

    supplied = object()
    seen = {}

    class _Recording(_StubSelector):
        def select_models(self, ticker, context_fingerprint, data=None):
            seen.update(data or {})
            return ["catboost"]

    manager = _Manager({"enabled_types": LIGHT})
    manager.context_selector = _Recording(None)

    manager._select_models_for_ticker("AAPL", {
        "context_fingerprint": "1|0",
        "current_context": pd.Series([9.0]),
        "similarity_finder": supplied,
    })

    assert seen["similarity_finder"] is supplied


def test_a_memory_layer_failure_does_not_stop_training():
    """The configured categories are a complete answer on their own."""
    class _Broken:
        def similarity_inputs(self, fingerprint, **kwargs):
            raise AttributeError("no database")

    manager = _Manager({"enabled_types": LIGHT}, recommendation=None,
                       history=_Broken())

    assert manager._select_models_for_ticker(
        "AAPL", {"context_fingerprint": "1|0"}
    ) == LIGHT


def test_the_heuristic_preference_matches_the_case_producers_use():
    """ModelFactory and models.yaml both spell it 'lstm', lower case.

    The check used to be `'LSTM' in self.available_models`, which no producer
    can satisfy, so the stated preference silently became "take the first
    entry" -- catboost, for the light category.
    """
    selector = ContextualModelSelector(["catboost", "lstm", "linear"])

    result = selector._heuristic_fallback("test")

    assert result["selected_model"] == "lstm"
    assert result["confidence"] == 0.0
    assert result["status"] == "Fallback"


def test_the_heuristic_falls_back_to_the_first_model_when_lstm_is_absent():
    selector = ContextualModelSelector(["catboost", "linear"])

    assert selector._heuristic_fallback("test")["selected_model"] == "catboost"
