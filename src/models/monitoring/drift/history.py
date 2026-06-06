import logging
from collections import deque
from datetime import datetime
from typing import Any

from src.core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)

class HistoryManager:
    """Керує історією прогнозів, продуктивності та стану дрейфу."""

    def __init__(self, window_size: int = 1000, reference_window_size: int = 5000):
        self.window_size = window_size
        self.reference_window_size = reference_window_size

        self.prediction_history = deque(maxlen=window_size)
        self.reference_predictions = deque(maxlen=reference_window_size)
        self.performance_history = deque(maxlen=1000)
        self.drift_history = []
        self.retraining_history = []

        self.logger = logger

    def update_prediction_history(self, predictions: Any, actuals: Any | None, confidences: Any | None, timestamp: datetime) -> None:
        """Оновлює історію прогнозів."""
        try:
            for i, pred in enumerate(predictions):
                record = {
                    'prediction': pred,
                    'actual': actuals[i] if actuals is not None and i < len(actuals) else None,
                    'confidence': confidences[i] if confidences is not None and i < len(confidences) else None,
                    'timestamp': timestamp
                }
                self.prediction_history.append(record)

            if len(self.reference_predictions) < self.reference_window_size:
                for record in list(self.prediction_history)[-len(predictions):]:
                    self.reference_predictions.append(record)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error updating prediction history: {e}")
            raise DataProcessingError("Failed to update prediction history") from e

    def add_performance_record(self, metrics: dict[str, Any], timestamp: datetime, sample_count: int):
        self.performance_history.append({
            'timestamp': timestamp,
            'metrics': metrics,
            'sample_count': sample_count
        })

    def add_drift_record(self, timestamp: datetime, drift_detected: bool, drift_severity: str, drift_score: float, methods: dict[str, Any]):
        self.drift_history.append({
            'timestamp': timestamp,
            'drift_detected': drift_detected,
            'drift_severity': drift_severity,
            'drift_score': drift_score,
            'methods': methods
        })
