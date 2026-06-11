import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from src.models.analysis.model_health_analyzer import ModelHealthAnalyzer

@pytest.mark.asyncio
async def test_model_health_analyzer_analyze():
    # Setup
    config = {
        "baseline_detector": {},
        "regime_analyzer": {},
        "overfitting_detector": {},
        "drift_monitor": {}
    }
    
    # We must patch the components in __init__ because they instantiate real objects
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.models.analysis.model_health_analyzer.BaselineDominanceDetector", MagicMock())
        m.setattr("src.models.analysis.model_health_analyzer.RegimeWinnerAnalyzer", MagicMock())
        m.setattr("src.models.analysis.model_health_analyzer.OverfittingDetector", MagicMock())
        m.setattr("src.models.analysis.model_health_analyzer.PredictionDriftMonitor", MagicMock())
        
        analyzer = ModelHealthAnalyzer(config)
        
        # Setup AsyncMocks for component methods
        analyzer.baseline_detector.analyze = AsyncMock(return_value={"status": "ok"})
        analyzer.regime_analyzer.analyze = AsyncMock(return_value={"status": "ok"})
        analyzer.overfitting_detector.analyze = AsyncMock(return_value={"status": "ok"})
        analyzer.drift_monitor.monitor = AsyncMock(return_value={"status": "ok"})
        
        # Test Data
        model = MagicMock()
        X = pd.DataFrame(np.random.rand(10, 5))
        y = pd.Series(np.random.rand(10))
        market_data = pd.DataFrame({'close': [1, 2, 3]})
        preds = np.array([0.1, 0.2])
        actuals = np.array([0.15, 0.25])
        conf = np.array([0.9, 0.9])
        
        # Execute
        result = await analyzer.analyze(
            model, "TestModel", X, y, 
            market_data=market_data, 
            predictions=preds, 
            actuals=actuals, 
            confidences=conf
        )
        
        # Assertions
        assert result["model_name"] == "TestModel"
        assert "analysis_results" in result
        assert result["analysis_results"]["baseline"]["status"] == "ok"
        assert result["analysis_results"]["drift"]["status"] == "ok"
        
        analyzer.baseline_detector.analyze.assert_awaited()
        analyzer.regime_analyzer.analyze.assert_awaited()
        analyzer.overfitting_detector.analyze.assert_awaited()
        analyzer.drift_monitor.monitor.assert_awaited()
