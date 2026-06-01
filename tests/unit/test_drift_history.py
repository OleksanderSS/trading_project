import pytest
import numpy as np
from datetime import datetime
from src.models.monitoring.drift.history import HistoryManager
from src.core.exceptions import DataProcessingError

def test_history_manager():
    manager = HistoryManager(window_size=10, reference_window_size=5)
    preds = np.array([0.1, 0.2])
    manager.update_prediction_history(preds, None, None, datetime.now())
    
    assert len(manager.prediction_history) == 2
    assert len(manager.reference_predictions) == 2

def test_history_manager_error():
    manager = HistoryManager()
    # Імітація помилки (наприклад, передача невалідних даних)
    with pytest.raises(DataProcessingError):
        manager.update_prediction_history(None, None, None, datetime.now())
