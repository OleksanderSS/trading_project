import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.models.analysis.overfitting_detection.manager import OverfittingDetector
from src.core.exceptions import DataProcessingError

@pytest.mark.asyncio
async def test_overfitting_detector_exception():
    """Перевірка підняття DataProcessingError при помилці аналізу."""
    # Створюємо фіктивну модель, яка викликає помилку при аналізі
    model = MagicMock()
    model.fit.side_effect = Exception("Model training failed")
    
    # Конфігурація для детектора
    config = {'enable_visualization': False}
    detector = OverfittingDetector(config)
    
    # Викликаємо аналіз з некоректними даними, щоб викликати помилку
    with pytest.raises(DataProcessingError, match="Overfitting detection failed"):
        await detector.detect_overfitting(model, pd.DataFrame(), pd.Series())
