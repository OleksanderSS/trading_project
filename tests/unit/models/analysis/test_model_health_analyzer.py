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
    
    # We must patch ModelAnalyzer in __init__ because it instantiates real components
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.models.analysis.model_health_analyzer.ModelAnalyzer", MagicMock())

        analyzer = ModelHealthAnalyzer(config)

        # Setup AsyncMocks matching ModelAnalyzer's real public method names
        analyzer.model_analyzer.perform_baseline_analysis = AsyncMock(return_value={"status": "ok"})
        analyzer.model_analyzer.perform_regime_analysis = AsyncMock(return_value={"status": "ok"})
        analyzer.model_analyzer.perform_overfitting_analysis = AsyncMock(return_value={"status": "ok"})
        analyzer.model_analyzer.perform_drift_monitoring = AsyncMock(return_value={"status": "ok"})
        
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
        
        analyzer.model_analyzer.perform_baseline_analysis.assert_awaited()
        analyzer.model_analyzer.perform_regime_analysis.assert_awaited()
        analyzer.model_analyzer.perform_overfitting_analysis.assert_awaited()
        analyzer.model_analyzer.perform_drift_monitoring.assert_awaited()
