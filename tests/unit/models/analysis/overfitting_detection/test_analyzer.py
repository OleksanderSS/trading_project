import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.models.analysis.overfitting_detection.analyzer import OverfittingAnalyzer

# Mocking config and metrics
@pytest.fixture
def analyzer():
    mock_config = MagicMock()
    mock_config.cv_folds = 3
    mock_config.train_sizes = np.linspace(0.1, 1.0, 3)
    mock_config.scoring_metric = 'neg_mean_squared_error'
    mock_config.thresholds = {'train_val_gap': {'threshold': 0.1}, 'cv_variance': {'threshold': 0.1}}
    
    mock_metrics = MagicMock()
    mock_metrics.calculate_metrics.return_value = {'rmse': 0.1, 'mse': 0.01}
    
    return OverfittingAnalyzer(mock_config, mock_metrics)

@pytest.mark.asyncio
@patch('src.models.analysis.overfitting_detection.analyzer.learning_curve')
async def test_generate_learning_curve(mock_lc, analyzer):
    # Mock learning_curve return values
    mock_lc.return_value = (
        np.array([10, 20, 30]), 
        np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]), 
        np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    )
    
    model = MagicMock()
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    
    result = await analyzer.generate_learning_curve(model, X, y)
    assert 'train_scores_mean' in result
    assert 'test_scores_mean' in result

@pytest.mark.asyncio
@patch('src.models.analysis.overfitting_detection.analyzer.cross_val_score')
async def test_perform_cv_analysis(mock_cv, analyzer):
    mock_cv.return_value = np.array([0.9, 0.8, 0.85])
    
    model = MagicMock()
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    
    result = await analyzer.perform_cv_analysis(model, X, y)
    assert 'mean' in result
    assert 'scores' in result

def test_analyze_train_val_gap(analyzer):
    model = MagicMock()
    model.predict.return_value = np.array([0.1, 0.2, 0.1, 0.2])
    X_train = pd.DataFrame(np.random.rand(4, 5))
    y_train = pd.Series([0.1, 0.2, 0.1, 0.2])
    X_val = pd.DataFrame(np.random.rand(4, 5))
    y_val = pd.Series([0.15, 0.25, 0.15, 0.25])
    
    result = analyzer.analyze_train_val_gap(model, X_train, y_train, X_val, y_val)
    assert 'gap' in result
    assert 'status' in result
