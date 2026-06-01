import pytest
import numpy as np
import pandas as pd
from src.models.analysis.overfitting_detection.metrics import OverfittingMetrics

def test_calculate_metrics():
    metrics = OverfittingMetrics()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    result = metrics.calculate_metrics(y_true, y_pred)
    
    assert 'mse' in result
    assert 'mae' in result
    assert 'r2' in result
    assert 'rmse' in result
    assert result['mse'] > 0
    assert result['rmse'] > 0

def test_calculate_metrics_invalid():
    metrics = OverfittingMetrics()
    # Invalid input (mismatched lengths) should trigger exception, handled by returning {}
    assert metrics.calculate_metrics(np.array([1, 2]), np.array([1])) == {}

def test_analyze_data_characteristics():
    metrics = OverfittingMetrics()
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    X_val = pd.DataFrame(np.random.rand(20, 5), columns=['a', 'b', 'c', 'd', 'e'])
    
    chars = metrics.analyze_data_characteristics(X_train, X_val)
    
    assert chars['n_train_samples'] == 100
    assert chars['n_features'] == 5
    assert chars['n_val_samples'] == 20
    assert chars['val_ratio'] == 0.2
