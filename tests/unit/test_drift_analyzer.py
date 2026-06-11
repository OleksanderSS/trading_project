import pytest
import numpy as np
from unittest.mock import MagicMock
from src.models.monitoring.drift.analyzer import DriftAnalyzer
from src.core.exceptions import DataProcessingError

@pytest.mark.asyncio
async def test_drift_analyzer_detection():
    # Мокаємо калькулятор дрейфу
    calculator = MagicMock()
    calculator.perform_ks_test.return_value = {'detected': True}
    calculator.calculate_psi.return_value = {'psi': 0.1}
    calculator.calculate_wasserstein_distance.return_value = 0.05
    calculator.calculate_overall_drift_score.return_value = 0.3
    calculator.determine_drift_severity.return_value = 'high'
    
    analyzer = DriftAnalyzer(calculator, min_samples=2)
    
    # Викликаємо аналіз
    current_preds = np.array([0.1, 0.2, 0.3])
    ref_preds = np.array([0.1, 0.15, 0.25])
    
    result = await analyzer.detect_prediction_drift(current_preds, ref_preds, None)
    
    assert result['drift_detected'] is True
    assert result['drift_severity'] == 'high'

@pytest.mark.asyncio
async def test_drift_analyzer_error():
    calculator = MagicMock()
    # Імітуємо помилку в калькуляторі, яку обробляє DriftAnalyzer
    calculator.perform_ks_test.side_effect = ValueError("KS test failed")

    analyzer = DriftAnalyzer(calculator, min_samples=2)

    with pytest.raises(DataProcessingError, match="Prediction drift detection failed"):
        await analyzer.detect_prediction_drift(np.array([0.1]), np.array([0.1, 0.2]), None)
