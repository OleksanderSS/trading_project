import logging
from typing import Any

import numpy as np

from src.core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)

class DriftAnalyzer:
    """Аналізує дрейф прогнозів, продуктивність та впевненість моделі."""

    def __init__(self, drift_calculator: Any, min_samples: int = 100):
        self.drift_calculator = drift_calculator
        self.min_samples = min_samples
        self.logger = logger

    async def detect_prediction_drift(self, current_predictions: np.ndarray, reference_predictions: np.ndarray, timestamp: Any) -> dict[str, Any]:
        """Виявляє дрейф розподілу прогнозів."""
        drift_analysis = {
            'status': 'completed',
            'drift_detected': False,
            'drift_methods': {},
            'overall_drift_score': 0.0,
            'drift_severity': 'none'
        }

        try:
            if len(reference_predictions) < self.min_samples:
                drift_analysis['status'] = 'insufficient_reference_data'
                return drift_analysis

            # Статистичні тести
            drift_analysis['drift_methods']['ks_test'] = self.drift_calculator.perform_ks_test(current_predictions, reference_predictions)
            drift_analysis['drift_methods']['psi'] = self.drift_calculator.calculate_psi(current_predictions, reference_predictions)
            drift_analysis['drift_methods']['wasserstein'] = self.drift_calculator.calculate_wasserstein_distance(current_predictions, reference_predictions)

            drift_analysis['overall_drift_score'] = self.drift_calculator.calculate_overall_drift_score(drift_analysis['drift_methods'])
            drift_analysis['drift_detected'] = drift_analysis['overall_drift_score'] > 0.2
            drift_analysis['drift_severity'] = self.drift_calculator.determine_drift_severity(drift_analysis['overall_drift_score'])

            return drift_analysis
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error detecting prediction drift: {e}")
            raise DataProcessingError("Prediction drift detection failed") from e

    def analyze_performance_degradation(self, actuals: np.ndarray, predictions: np.ndarray, performance_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Аналізує деградацію продуктивності."""
        try:
            current_metrics = self.drift_calculator.calculate_performance_metrics(actuals, predictions)

            performance_analysis = {
                'current_performance': current_metrics,
                'performance_trend': {},
                'degradation_detected': False,
                'degradation_score': 0.0
            }

            if len(performance_history) >= 10:
                trend_analysis = self.drift_calculator.analyze_performance_trend(performance_history)
                performance_analysis['performance_trend'] = trend_analysis
                degradation_score = trend_analysis.get('degradation_score', 0.0)
                performance_analysis['degradation_detected'] = degradation_score > 0.1
                performance_analysis['degradation_score'] = degradation_score

            return performance_analysis
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error analyzing performance degradation: {e}")
            raise DataProcessingError("Performance degradation analysis failed") from e

    def analyze_confidence_drift(self, confidences: np.ndarray, reference_confidences: np.ndarray) -> dict[str, Any]:
        """Аналізує дрейф впевненості моделі."""
        try:
            confidence_analysis = {
                'current_confidence_stats': self.drift_calculator.calculate_confidence_stats(confidences),
                'confidence_drift_detected': False,
                'drift_score': 0.0
            }

            if len(reference_confidences) >= self.min_samples:
                drift_result = self.drift_calculator.calculate_confidence_drift(confidences, reference_confidences)
                confidence_analysis['confidence_drift_detected'] = drift_result['drift_detected']
                confidence_analysis['drift_score'] = drift_result['drift_score']
                confidence_analysis['reference_stats'] = drift_result['reference_stats']

            return confidence_analysis
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error analyzing confidence drift: {e}")
            raise DataProcessingError("Confidence drift analysis failed") from e
