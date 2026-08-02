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


class _Manager:
    """Exercises the selection logic without building trainers or an arena."""

    def __init__(self, models_config, recommendation=None):
        self._models_config = models_config
        self.logger = __import__("logging").getLogger("test")
        self.context_selector = _StubSelector(recommendation)

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
