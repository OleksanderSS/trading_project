from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .base import CalibrationStrategy


class PlattScalingStrategy(CalibrationStrategy):
    """Platt scaling (logistic regression) для бінарної класифікації."""

    def __init__(self):
        self.calibrator = LogisticRegression(random_state=42)

    def fit(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> dict[str, Any]:
        try:
            scores = predictions[:, 1].reshape(-1, 1) if predictions.ndim == 2 else predictions.reshape(-1, 1)
            self.calibrator.fit(scores, targets)
            return {'calibrator_type': 'platt_scaling', 'coefficients': self.calibrator.coef_.tolist()}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.handle_error(e, "Platt scaling")

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        scores = predictions[:, 1].reshape(-1, 1) if predictions.ndim == 2 else predictions.reshape(-1, 1)
        return self.calibrator.predict_proba(scores)[:, 1]

class IsotonicRegressionStrategy(CalibrationStrategy):
    """Isotonic regression для калібрування ймовірностей."""

    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.calibrator = None

    def fit(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> dict[str, Any]:
        try:
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(predictions, targets)
            return {'calibrator_type': 'isotonic_regression'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.handle_error(e, "Isotonic regression")

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(predictions)
