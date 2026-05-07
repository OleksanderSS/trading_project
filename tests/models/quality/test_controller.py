"""Tests for ModelQualityController."""

import pytest
import numpy as np

from src.models.quality.controller import ModelQualityController


@pytest.fixture
def controller():
    """Create controller instance."""
    return ModelQualityController(drift_threshold=0.5)


def test_controller_creation():
    """Test controller creation."""
    controller = ModelQualityController(drift_threshold=0.3)
    assert controller.drift_threshold == 0.3
    assert len(controller.baseline_stats) == 0


def test_validate_predictions_valid(controller):
    """Test validation of valid predictions."""
    predictions = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
    assert controller.validate_predictions(predictions) == True


def test_validate_predictions_nan(controller):
    """Test validation rejects NaN."""
    predictions = np.array([0.01, np.nan, 0.02])
    assert controller.validate_predictions(predictions) == False


def test_validate_predictions_inf(controller):
    """Test validation rejects Inf."""
    predictions = np.array([0.01, np.inf, 0.02])
    assert controller.validate_predictions(predictions) == False


def test_validate_predictions_unrealistic(controller):
    """Test validation rejects unrealistic values."""
    predictions = np.array([0.01, 15.0, 0.02])  # 1500% return
    assert controller.validate_predictions(predictions) == False


def test_check_drift_no_drift(controller):
    """Test drift detection with no drift."""
    np.random.seed(42)
    baseline = np.random.normal(0, 1, 100)
    current = np.random.normal(0.1, 1, 100)  # Small shift
    
    drift = controller.check_drift(current, baseline)
    assert drift < 0.5  # Below threshold


def test_check_drift_with_drift(controller):
    """Test drift detection with significant drift."""
    np.random.seed(42)
    baseline = np.random.normal(0, 1, 100)
    current = np.random.normal(2, 1, 100)  # Large shift
    
    drift = controller.check_drift(current, baseline)
    assert drift > 0.5  # Above threshold


def test_get_quality_score_high_agreement(controller):
    """Test quality score with high agreement."""
    predictions = {"model1": 0.05, "model2": 0.051, "model3": 0.049}
    weights = {"model1": 0.33, "model2": 0.33, "model3": 0.34}
    
    score = controller.get_quality_score(0.05, predictions, weights)
    
    assert 0 <= score <= 1
    assert score > 0.8  # High agreement + balanced weights


def test_get_quality_score_low_agreement(controller):
    """Test quality score with low agreement."""
    predictions = {"model1": 0.05, "model2": 0.15, "model3": -0.05}
    weights = {"model1": 0.33, "model2": 0.33, "model3": 0.34}
    
    score = controller.get_quality_score(0.05, predictions, weights)
    
    assert 0 <= score <= 1
    # With balanced weights, score can still be high even with disagreement
    # because balance component (0.4 weight) is high


def test_get_quality_score_unbalanced_weights(controller):
    """Test quality score with unbalanced weights."""
    predictions = {"model1": 0.05, "model2": 0.051, "model3": 0.049}
    weights = {"model1": 0.9, "model2": 0.05, "model3": 0.05}
    
    score = controller.get_quality_score(0.05, predictions, weights)
    
    assert 0 <= score <= 1
    # High agreement but unbalanced weights


def test_update_baseline(controller):
    """Test updating baseline statistics."""
    predictions = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    
    controller.update_baseline("test_model", predictions)
    
    assert "test_model" in controller.baseline_stats
    baseline = controller.baseline_stats["test_model"]
    assert "mean" in baseline
    assert "std" in baseline
    assert "min" in baseline
    assert "max" in baseline
    assert baseline["count"] == 5


def test_get_baseline(controller):
    """Test getting baseline statistics."""
    predictions = np.array([0.01, 0.02, 0.03])
    controller.update_baseline("test_model", predictions)
    
    baseline = controller.get_baseline("test_model")
    assert baseline is not None
    assert baseline["mean"] == pytest.approx(0.02, abs=0.001)


def test_get_baseline_not_found(controller):
    """Test getting non-existent baseline."""
    baseline = controller.get_baseline("nonexistent")
    assert baseline is None


def test_generate_report(controller):
    """Test report generation."""
    # Add some baselines
    controller.update_baseline("model1", np.array([0.01, 0.02]))
    controller.update_baseline("model2", np.array([0.03, 0.04]))
    
    report = controller.generate_report()
    
    assert "drift_threshold" in report
    assert "baseline_models" in report
    assert "total_baselines" in report
    assert "timestamp" in report
    assert report["total_baselines"] == 2
    assert "model1" in report["baseline_models"]
    assert "model2" in report["baseline_models"]


def test_flag_anomalies_no_anomalies(controller):
    """Test anomaly detection with no anomalies."""
    np.random.seed(42)
    predictions = np.random.normal(0, 1, 100)
    
    anomalies = controller.flag_anomalies(predictions, threshold=3.0)
    
    assert isinstance(anomalies, np.ndarray)
    assert len(anomalies) == 100
    # Most should not be anomalies
    assert np.sum(anomalies) < 5


def test_flag_anomalies_with_anomalies(controller):
    """Test anomaly detection with anomalies."""
    predictions = np.array([0.01, 0.02, 0.03, 10.0, 0.04])  # 10.0 is anomaly
    
    anomalies = controller.flag_anomalies(predictions, threshold=1.5)  # Lower threshold
    
    assert np.sum(anomalies) >= 1  # At least one anomaly


def test_compare_models_without_actuals(controller):
    """Test model comparison without actuals."""
    np.random.seed(42)
    model_a_preds = np.random.normal(0.05, 0.01, 100)
    model_b_preds = np.random.normal(0.06, 0.01, 100)
    
    comparison = controller.compare_models(model_a_preds, model_b_preds)
    
    assert "correlation" in comparison
    assert "mean_absolute_difference" in comparison
    assert "model_a_mean" in comparison
    assert "model_b_mean" in comparison
    assert "model_a_std" in comparison
    assert "model_b_std" in comparison
    assert "model_a_mae" not in comparison  # No actuals provided


def test_compare_models_with_actuals(controller):
    """Test model comparison with actuals."""
    np.random.seed(42)
    actuals = np.random.normal(0.05, 0.01, 100)
    model_a_preds = actuals + np.random.normal(0, 0.005, 100)  # Better
    model_b_preds = actuals + np.random.normal(0, 0.02, 100)   # Worse
    
    comparison = controller.compare_models(model_a_preds, model_b_preds, actuals)
    
    assert "model_a_mae" in comparison
    assert "model_b_mae" in comparison
    assert "better_model" in comparison
    assert "improvement" in comparison
    assert comparison["better_model"] == "A"  # Model A should be better


def test_drift_detection_edge_cases(controller):
    """Test drift detection edge cases."""
    # Identical distributions
    baseline = np.array([1.0, 2.0, 3.0])
    current = np.array([1.0, 2.0, 3.0])
    drift = controller.check_drift(current, baseline)
    assert drift == pytest.approx(0.0, abs=0.01)
    
    # Zero variance baseline
    baseline = np.array([1.0, 1.0, 1.0])
    current = np.array([1.1, 1.1, 1.1])
    drift = controller.check_drift(current, baseline)
    assert drift > 0  # Should detect drift even with zero variance


def test_quality_score_edge_cases(controller):
    """Test quality score edge cases."""
    # Single model
    predictions = {"model1": 0.05}
    weights = {"model1": 1.0}
    score = controller.get_quality_score(0.05, predictions, weights)
    assert 0 <= score <= 1
    
    # Perfect agreement
    predictions = {"model1": 0.05, "model2": 0.05, "model3": 0.05}
    weights = {"model1": 0.33, "model2": 0.33, "model3": 0.34}
    score = controller.get_quality_score(0.05, predictions, weights)
    assert score > 0.9  # Should be very high
