from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from src.core.exceptions import DataProcessingError

class CalibrationStrategy(ABC):
    """Базовий клас для стратегій калібрування впевненості."""

    @abstractmethod
    def fit(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Навчання стратегії."""
        pass

    @abstractmethod
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Застосування калібрування."""
        pass

    def handle_error(self, e: Exception, operation: str) -> None:
        raise DataProcessingError(f"Calibration error during {operation}: {e}") from e
