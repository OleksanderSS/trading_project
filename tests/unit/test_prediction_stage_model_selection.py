import logging

import pandas as pd

from src.pipeline.stages.stage_5_prediction import PredictionStage


class StubSmartSelector:
    def __init__(self, selected_model_type: str):
        self.selected_model_type = selected_model_type
        self.available_models = None

    def select_best_model(self, df, target_type, available_models):
        self.available_models = available_models
        return self.selected_model_type, 0.9


def _stage_with_selector(selector):
    stage = object.__new__(PredictionStage)
    stage.context_selector = selector
    stage.logger = logging.getLogger("test_prediction_stage")
    return stage


def test_context_model_selection_excludes_autoencoder_and_resolves_model_key():
    selector = StubSmartSelector("lightgbm")
    stage = _stage_with_selector(selector)
    features = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    models = {
        "model_AAPL_target_return_1d_lightgbm": object(),
        "model_AAPL_target_return_1d_catboost": object(),
        "model_AAPL_target_return_1d_autoencoder": object(),
    }

    selected = stage._select_best_model_for_context(
        features,
        {"target_type": "regression"},
        models,
        "AAPL",
        "bull",
    )

    assert selected == "model_AAPL_target_return_1d_lightgbm"
    assert selector.available_models == ["lightgbm", "catboost"]


def test_prediction_stage_no_longer_exposes_dead_knn_selection_path():
    assert not hasattr(PredictionStage, "_perform_knn_similarity_analysis")
    assert not hasattr(PredictionStage, "_analyze_knn_similarities")
