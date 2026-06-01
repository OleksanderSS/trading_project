import pandas as pd

from src.training.pattern_aware_training import PatternAwareModelTrainer, train_pattern_aware_models


def test_pattern_aware_training_trains_and_selects_champion():
    features = pd.DataFrame({
        "close": [100, 101, 102, 103, 104, 105, 106, 107],
        "momentum": [0.1, 0.2, 0.15, 0.25, 0.3, 0.28, 0.35, 0.4],
    })
    targets = pd.DataFrame({
        "target_return": [0.01, 0.02, 0.015, 0.025, 0.03, 0.028, 0.035, 0.04],
    })
    trainer = PatternAwareModelTrainer(config={"model_names": ["linear", "ridge"]})

    result = trainer.train_pattern_aware_models(features, targets, patterns={"regime": "normal"})

    assert result["status"] == "success"
    assert result["best_model"]["model_name"] in {"linear", "ridge"}
    assert "model_metrics" in result


def test_pattern_aware_training_skips_without_data():
    result = train_pattern_aware_models()

    assert result == {"status": "skipped", "reason": "missing_training_data"}


def test_pattern_aware_training_uses_volatile_parameters():
    trainer = PatternAwareModelTrainer()

    params = trainer._get_adaptive_parameters("random_forest", {"regime": "volatile"})

    assert params["max_depth"] == 4
    assert params["min_samples_leaf"] == 3
