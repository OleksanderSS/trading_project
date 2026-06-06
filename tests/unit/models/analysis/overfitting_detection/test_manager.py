import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from src.models.analysis.overfitting_detection.manager import OverfittingDetector
from src.core.exceptions import DataProcessingError

@pytest.fixture
def detector():
    return OverfittingDetector()

@pytest.mark.asyncio
async def test_detect_overfitting_success(detector):
    model = MagicMock()
    model.__class__.__name__ = "TestModel"
    X = pd.DataFrame(np.random.rand(10, 2))
    y = pd.Series(np.random.rand(10))
    
    # Mock internal components using AsyncMock
    detector.analyzer.generate_learning_curve = AsyncMock(return_value={})
    detector.analyzer.perform_cv_analysis = AsyncMock(return_value={})
    detector.analyzer.detect_signals = MagicMock(return_value={})
    detector.analyzer.generate_recommendations = MagicMock(return_value=[])
    
    # Mock store and viz
    detector._store_results = MagicMock()
    
    result = await detector.detect_overfitting(model, X, y)
    
    assert result['model_type'] == "TestModel"
    assert 'learning_curve' in result
    detector._store_results.assert_called_once()

@pytest.mark.asyncio
async def test_detect_overfitting_failure(detector):
    model = MagicMock()
    X = pd.DataFrame(np.random.rand(10, 2))
    y = pd.Series(np.random.rand(10))
    
    # Force exception in analyzer
    detector.analyzer.generate_learning_curve = AsyncMock(side_effect=ValueError("Analysis failed"))
    
    with pytest.raises(DataProcessingError, match="Analysis failed"):
        await detector.detect_overfitting(model, X, y)

def test_get_overfitting_summary(detector):
    # Empty history
    summary = detector.get_overfitting_summary()
    assert summary['total_analyses'] == 0
    
    # Mock history
    detector.history = [{'model_type': 'ModelA', 'timestamp': '2026-05-31T00:00:00'}]
    summary = detector.get_overfitting_summary()
    assert summary['total_analyses'] == 1
    assert 'ModelA' in summary['models_analyzed']
