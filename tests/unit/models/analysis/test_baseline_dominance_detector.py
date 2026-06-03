import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from src.models.analysis.baseline_dominance_detector import BaselineDominanceDetector
from src.core.exceptions import DataProcessingError

@pytest.fixture
def detector():
    # Patch init_components to avoid complex dependencies
    with pytest.MonkeyPatch.context() as m:
        m.setattr(BaselineDominanceDetector, "_init_components", MagicMock())
        detector = BaselineDominanceDetector()
        # Setup mocks for engine/recommendations manually
        detector.comparison_engine = MagicMock()
        detector.recommendation_engine = MagicMock()
        detector.baseline_implementations = {}
        return detector

@pytest.mark.asyncio
async def test_analyze_baseline_dominance_success(detector):
    complex_results = {'metrics': {'rmse': 0.05}}
    market_data = pd.DataFrame({'close': [1, 2, 3]})
    
    # Mock _train_baseline_models
    detector._train_baseline_models = AsyncMock(return_value={'model1': {'score': 0.1}})
    
    # Mock engines
    detector.comparison_engine.compare.return_value = {'dominance': False}
    detector.recommendation_engine.perform_cost_benefit_analysis.return_value = {}
    detector.recommendation_engine.generate_simplification_recommendations.return_value = []
    
    result = await detector.analyze_baseline_dominance(complex_results, market_data)
    
    assert 'dominance_analysis' in result
    assert result['dominance_analysis'] == {'dominance': False}
    detector._train_baseline_models.assert_awaited_once()

@pytest.mark.asyncio
async def test_analyze_baseline_dominance_failure(detector):
    complex_results = {'metrics': {'rmse': 0.05}}
    market_data = pd.DataFrame({'close': [1, 2, 3]})
    
    # Force exception
    detector._train_baseline_models = AsyncMock(side_effect=Exception("Training failed"))
    
    with pytest.raises(DataProcessingError):
        await detector.analyze_baseline_dominance(complex_results, market_data)
