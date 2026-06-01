import logging

from src.training.unified_training_manager import UnifiedTrainingManager, TrainingStrategy


class StubConfigManager:
    def __init__(self, models_config=None):
        self.models_config = models_config or {}

    def get_config(self, key, default=None):
        if key == "models":
            return self.models_config
        return default


class RaisingSelector:
    def select_models(self, ticker, context_fingerprint):
        raise RuntimeError("selector unavailable")


def _manager(config):
    manager = object.__new__(UnifiedTrainingManager)
    manager.config_manager = config
    manager.context_selector = object()
    manager.logger = logging.getLogger("test_unified_training_manager")
    return manager


def test_select_models_falls_back_to_factory_models_when_config_missing(monkeypatch):
    manager = _manager(StubConfigManager())
    monkeypatch.setattr(manager, "_get_available_model_names", lambda: ["Linear", "LightGBM"])

    assert manager._select_models_for_ticker("AAPL", {}) == ["Linear", "LightGBM"]


def test_select_models_uses_configured_models_after_selector_failure():
    manager = _manager(StubConfigManager({"enabled_types": ["lightgbm", "catboost"]}))
    manager.context_selector = RaisingSelector()

    assert manager._select_models_for_ticker("AAPL", {"context_fingerprint": "ctx"}) == [
        "lightgbm",
        "catboost",
    ]


class FailingArena:
    def run_battle(self, tickers_results, actual_targets):
        raise RuntimeError("arena offline")


class StubBatchTrainer:
    def execute_batch_training(self, plan, data_context):
        return {"tickers_results": {"AAPL": {"status": "trained"}}}


def test_execute_unified_training_keeps_training_results_when_arena_fails():
    manager = _manager(StubConfigManager())
    manager.arena = FailingArena()
    manager.trainers = {TrainingStrategy.BATCH.value: StubBatchTrainer()}
    manager.create_unified_plan = lambda tickers: {
        "strategy": TrainingStrategy.BATCH.value,
        "ticker_plans": {},
    }
    manager.save_unified_plan = lambda plan: None
    manager.save_unified_results = lambda results: None
    manager._select_models_for_ticker = lambda ticker, data_context: ["lightgbm"]

    result = manager.execute_unified_training(["AAPL"], {"y_test": [1]})

    assert result["tickers_results"] == {"AAPL": {"status": "trained"}}
    assert result["arena_error"] == "arena offline"
