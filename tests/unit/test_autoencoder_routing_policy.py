import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataProcessingError
from src.pipeline.hybrid.final_stages_executor import FinalStagesExecutor
from src.predictions.models_predict import get_predictions, predict_any


class _Predictor:
    def predict(self, X):
        return np.ones(len(X), dtype=float)


class _AutoencoderShouldNotPredict:
    def predict(self, X):
        raise AssertionError("autoencoder should not be used as target predictor")


def test_final_stages_executor_does_not_treat_autoencoder_as_heavy_predictor():
    executor = FinalStagesExecutor(config_manager=None, output_dir="data", batch_name="test")
    colab_results = {
        "ticker_results": {
            "AAA": {
                "timeframes": {
                    "all": {
                        "results": {
                            "target_return_1d": {
                                "models": {"Autoencoder": {"status": "success"}}
                            }
                        }
                    }
                }
            }
        }
    }

    assert executor._has_heavy_models(colab_results) is False


def test_final_stages_executor_train_heavy_models_excludes_autoencoder():
    executor = FinalStagesExecutor(config_manager=None, output_dir="data", batch_name="test")
    features = pd.DataFrame({"f1": [1.0, 2.0]})
    targets = pd.DataFrame({"target_return_1d": [0.1, -0.1]})

    result = asyncio.run(executor._train_heavy_models(features, targets, ["AAA"]))
    models = result["ticker_results"]["AAA"]["timeframes"]["all"]["results"][
        "target_return_1d"
    ]["models"]

    assert "autoencoder" not in models
    assert set(models) == {"cnn", "lstm", "gru", "transformer", "tabnet"}


def test_predict_any_rejects_autoencoder_as_target_predictor():
    with pytest.raises(DataProcessingError, match="reserved for anomaly"):
        predict_any(_AutoencoderShouldNotPredict(), np.ones((3, 2)), "autoencoder")


def test_get_predictions_skips_autoencoder_models(monkeypatch):
    def fake_ensemble_forecast(model_predictions, **kwargs):
        assert set(model_predictions) == {"linear"}
        return SimpleNamespace(final_signal=np.array([1.0, 1.0]), stats={"ok": True})

    monkeypatch.setattr(
        "src.predictions.models_predict.ensemble_forecast",
        fake_ensemble_forecast,
    )

    result = get_predictions(
        {
            "linear": _Predictor(),
            "autoencoder": _AutoencoderShouldNotPredict(),
        },
        pd.DataFrame({"f1": [1.0, 2.0]}),
    )

    assert "linear" in result
    assert "autoencoder" not in result
    assert "ensemble" in result
