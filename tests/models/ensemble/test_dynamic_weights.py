"""Tests for DynamicWeightCalculator."""

import pytest
import tempfile
import json
from pathlib import Path

from src.models.ensemble.dynamic_weights import DynamicWeightCalculator


@pytest.fixture
def calculator():
    """Create calculator instance."""
    return DynamicWeightCalculator(method="adaptive", alpha=0.3)


def test_calculator_creation():
    """Test calculator creation."""
    calc = DynamicWeightCalculator(method="performance_based", alpha=0.5)
    assert calc.method == "performance_based"
    assert calc.alpha == 0.5
    assert len(calc.weight_history) == 0


def test_performance_based_weights(calculator):
    """Test performance-based weight calculation."""
    models = ["model1", "model2", "model3"]
    performance = {"model1": 0.8, "model2": 0.6, "model3": 0.9}
    
    calculator.method = "performance_based"
    weights = calculator.calculate_weights(models, performance)
    
    # Check weights sum to 1
    assert abs(sum(weights.values()) - 1.0) < 0.001
    
    # Check model3 has highest weight (best performance)
    assert weights["model3"] > weights["model1"]
    assert weights["model1"] > weights["model2"]


def test_context_aware_weights(calculator):
    """Test context-aware weight calculation."""
    models = ["catboost", "lstm", "xgboost"]
    performance = {"catboost": 0.8, "lstm": 0.7, "xgboost": 0.8}
    context = {"volatility": 0.8, "trend": 0.1}
    
    calculator.method = "context_aware"
    weights = calculator.calculate_weights(models, performance, context)
    
    # Check weights sum to 1
    assert abs(sum(weights.values()) - 1.0) < 0.001
    
    # LSTM should get boost in high volatility
    assert weights["lstm"] > 0.2


def test_adaptive_weights(calculator):
    """Test adaptive weight calculation with history."""
    models = ["model1", "model2"]
    performance = {"model1": 0.8, "model2": 0.6}
    
    # First calculation
    weights1 = calculator.calculate_weights(models, performance)
    
    # Second calculation with different performance
    performance2 = {"model1": 0.6, "model2": 0.8}
    weights2 = calculator.calculate_weights(models, performance2)
    
    # Weights should be smoothed (not jump completely)
    # model1 weight should decrease but not to minimum
    assert weights2["model1"] < weights1["model1"]
    assert weights2["model1"] > 0.3  # Smoothed, not minimum


def test_equal_weights_fallback(calculator):
    """Test equal weights when no performance data."""
    models = ["model1", "model2", "model3"]
    performance: dict[str, float] = {}
    
    calculator.method = "performance_based"
    weights = calculator.calculate_weights(models, performance)
    
    # Should give equal weights
    assert abs(weights["model1"] - 1/3) < 0.001
    assert abs(weights["model2"] - 1/3) < 0.001
    assert abs(weights["model3"] - 1/3) < 0.001


def test_weight_history_tracking(calculator):
    """Test weight history is tracked."""
    models = ["model1", "model2"]
    performance = {"model1": 0.8, "model2": 0.6}
    
    # Calculate weights multiple times
    for _ in range(5):
        calculator.calculate_weights(models, performance)
    
    # Check history
    history1 = calculator.get_weight_history("model1")
    assert len(history1) == 5
    
    history2 = calculator.get_weight_history("model2")
    assert len(history2) == 5


def test_get_weight_history_window(calculator):
    """Test getting weight history with window."""
    models = ["model1"]
    performance = {"model1": 0.8}
    
    # Calculate 10 times
    for _ in range(10):
        calculator.calculate_weights(models, performance)
    
    # Get last 3
    history = calculator.get_weight_history("model1", window=3)
    assert len(history) == 3


def test_export_import_weights(calculator):
    """Test exporting and importing weights."""
    models = ["model1", "model2"]
    performance = {"model1": 0.8, "model2": 0.6}
    
    # Calculate some weights
    for _ in range(3):
        calculator.calculate_weights(models, performance)
    
    # Export
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    calculator.export_weights(filepath)
    
    # Create new calculator and import
    new_calc = DynamicWeightCalculator()
    new_calc.import_weights(filepath)
    
    # Check history preserved
    assert len(new_calc.weight_history["model1"]) == 3
    assert new_calc.method == calculator.method
    assert new_calc.alpha == calculator.alpha
    
    # Cleanup
    Path(filepath).unlink()


def test_get_stats(calculator):
    """Test getting statistics."""
    models = ["model1", "model2", "model3"]
    performance = {"model1": 0.8, "model2": 0.6, "model3": 0.7}
    
    # Calculate weights
    for _ in range(5):
        calculator.calculate_weights(models, performance)
    
    stats = calculator.get_stats()
    
    assert stats['method'] == 'adaptive'
    assert stats['alpha'] == 0.3
    assert stats['models_tracked'] == 3
    assert stats['total_calculations'] == 15  # 3 models * 5 calculations
    assert 'average_weights' in stats


def test_reset_history(calculator):
    """Test resetting history."""
    models = ["model1", "model2"]
    performance = {"model1": 0.8, "model2": 0.6}
    
    # Calculate weights
    calculator.calculate_weights(models, performance)
    assert len(calculator.weight_history) > 0
    
    # Reset
    calculator.reset_history()
    assert len(calculator.weight_history) == 0


def test_context_adjustments_lstm(calculator):
    """Test LSTM gets boost in high volatility."""
    models = ["lstm", "catboost"]
    performance = {"lstm": 0.7, "catboost": 0.7}  # Equal performance
    
    calculator.method = "context_aware"
    
    # High volatility
    context_high_vol = {"volatility": 0.9, "trend": 0.0}
    weights_high = calculator.calculate_weights(models, performance, context_high_vol)
    
    # Low volatility
    context_low_vol = {"volatility": 0.2, "trend": 0.0}
    weights_low = calculator.calculate_weights(models, performance, context_low_vol)
    
    # LSTM should have higher weight in high volatility
    assert weights_high["lstm"] > weights_low["lstm"]


def test_context_adjustments_tree(calculator):
    """Test tree models get boost in trending markets."""
    models = ["catboost", "lstm"]
    performance = {"catboost": 0.7, "lstm": 0.7}  # Equal performance
    
    calculator.method = "context_aware"
    
    # Strong trend
    context_trend = {"volatility": 0.5, "trend": 0.8}
    weights_trend = calculator.calculate_weights(models, performance, context_trend)
    
    # No trend
    context_no_trend = {"volatility": 0.5, "trend": 0.0}
    weights_no_trend = calculator.calculate_weights(models, performance, context_no_trend)
    
    # CatBoost should have higher weight in trending market
    assert weights_trend["catboost"] > weights_no_trend["catboost"]


def test_weights_always_sum_to_one(calculator):
    """Test weights always sum to 1.0."""
    models = ["model1", "model2", "model3"]
    performance = {"model1": 0.8, "model2": 0.6, "model3": 0.9}
    
    for method in ["performance_based", "context_aware", "adaptive"]:
        calculator.method = method
        weights = calculator.calculate_weights(models, performance)
        assert abs(sum(weights.values()) - 1.0) < 0.001


def test_adaptive_smoothing_factor(calculator):
    """Test adaptive smoothing with different alpha values."""
    models = ["model1", "model2"]
    
    # High alpha (more reactive)
    calc_high = DynamicWeightCalculator(method="adaptive", alpha=0.9)
    calc_high.calculate_weights(models, {"model1": 0.8, "model2": 0.2})
    weights_high = calc_high.calculate_weights(models, {"model1": 0.2, "model2": 0.8})
    
    # Low alpha (more stable)
    calc_low = DynamicWeightCalculator(method="adaptive", alpha=0.1)
    calc_low.calculate_weights(models, {"model1": 0.8, "model2": 0.2})
    weights_low = calc_low.calculate_weights(models, {"model1": 0.2, "model2": 0.8})
    
    # High alpha should change more (model1 weight should drop more)
    assert weights_high["model1"] < weights_low["model1"]
