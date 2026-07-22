#!/usr/bin/env python3
"""
Drift Calculator - Statistical drift detection methods
Handles mathematical calculations for prediction drift detection.
"""

from typing import Any

import numpy as np
from scipy import stats

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DriftCalculator")


class DriftCalculator:
    """
    Statistical drift detection calculator.

    Performs mathematical calculations for drift detection:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Wasserstein distance
    - Performance degradation analysis
    - Confidence drift analysis
    """

    def __init__(self, thresholds: dict[str, Any] | None = None):
        """
        Initialize Drift Calculator.

        Args:
            thresholds: Dictionary of drift detection thresholds
        """
        self.logger = logger

        # Default thresholds
        self.thresholds = thresholds or {
            'ks_test': {'threshold': 0.05, 'severity': 'high'},
            'psi': {'threshold': 0.25, 'severity': 'medium'},
            'wasserstein': {'threshold': 0.1, 'severity': 'medium'},
            'performance_degradation': {'threshold': 0.1, 'severity': 'high'},
            'confidence_drift': {'threshold': 0.15, 'severity': 'medium'}
        }

        self.logger.info("✅ DriftCalculator initialized")

    def perform_ks_test(self,
                       current_predictions: np.ndarray,
                       reference_predictions: np.ndarray) -> dict[str, Any]:
        """
        Perform Kolmogorov-Smirnov test for drift detection.
        """
        try:
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(current_predictions, reference_predictions)

            # Determine drift based on p-value
            threshold = self.thresholds['ks_test']['threshold']
            drift_detected = p_value < threshold

            # Calculate drift score (1 - p_value, normalized)
            drift_score = 1.0 - p_value

            return {
                'method': 'ks_test',
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'threshold': threshold,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'severity': self.thresholds['ks_test']['severity'] if drift_detected else 'none'
            }

        except Exception as e:
            self.logger.error(f"Error performing KS test: {e}", exc_info=True)
            raise RuntimeError(f"KS test calculation failed: {e}") from e

    def calculate_psi(self,
                     current_predictions: np.ndarray,
                     reference_predictions: np.ndarray,
                     bins: int = 10) -> dict[str, Any]:
        """
        Calculate Population Stability Index (PSI).
        """
        try:
            # Create bins based on reference distribution
            min_val = min(np.min(reference_predictions), np.min(current_predictions))
            max_val = max(np.max(reference_predictions), np.max(current_predictions))

            # Create bins
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            # Calculate frequencies
            ref_counts, _ = np.histogram(reference_predictions, bins=bin_edges)
            curr_counts, _ = np.histogram(current_predictions, bins=bin_edges)

            # Convert to percentages
            ref_percents = ref_counts / len(reference_predictions)
            curr_percents = curr_counts / len(current_predictions)

            # Calculate PSI
            psi = 0.0
            for i in range(len(ref_percents)):
                if ref_percents[i] > 0:  # Avoid division by zero
                    if curr_percents[i] == 0:
                        curr_percents[i] = 0.0001  # Small value to avoid log(0)

                    psi += (curr_percents[i] - ref_percents[i]) * np.log(curr_percents[i] / ref_percents[i])

            # Determine drift based on PSI threshold
            threshold = self.thresholds['psi']['threshold']
            drift_detected = psi > threshold

            # Normalize PSI to 0-1 scale
            drift_score = min(psi / (threshold * 2), 1.0)

            return {
                'method': 'psi',
                'psi_value': psi,
                'threshold': threshold,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'severity': self.thresholds['psi']['severity'] if drift_detected else 'none',
                'bins': bins,
                'ref_distribution': ref_percents.tolist(),
                'curr_distribution': curr_percents.tolist()
            }

        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Error calculating PSI: {e}", exc_info=True)
            raise RuntimeError(f"PSI calculation failed: {e}") from e

    def calculate_wasserstein_distance(self,
                                      current_predictions: np.ndarray,
                                      reference_predictions: np.ndarray) -> dict[str, Any]:
        """
        Calculate Wasserstein distance for drift detection.
        """
        try:
            # Calculate Wasserstein distance
            from scipy.stats import wasserstein_distance
            distance = wasserstein_distance(current_predictions, reference_predictions)

            # Normalize distance (rough normalization based on data range)
            data_range = np.max(reference_predictions) - np.min(reference_predictions)
            if data_range > 0:
                normalized_distance = distance / data_range
            else:
                normalized_distance = 0.0

            # Determine drift based on threshold
            threshold = self.thresholds['wasserstein']['threshold']
            drift_detected = normalized_distance > threshold

            # Calculate drift score
            drift_score = min(normalized_distance / (threshold * 2), 1.0)

            return {
                'method': 'wasserstein',
                'distance': distance,
                'normalized_distance': normalized_distance,
                'threshold': threshold,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'severity': self.thresholds['wasserstein']['severity'] if drift_detected else 'none'
            }

        except Exception as e:
            self.logger.error(f"Error calculating Wasserstein distance: {e}", exc_info=True)
            raise RuntimeError(f"Wasserstein distance calculation failed: {e}") from e

    def calculate_performance_metrics(self,
                                    actuals: np.ndarray,
                                    predictions: np.ndarray) -> dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            actuals: Actual values
            predictions: Predicted values

        Returns:
            Dictionary with performance metrics
        """
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            metrics = {
                'mse': mean_squared_error(actuals, predictions),
                'mae': mean_absolute_error(actuals, predictions),
                'r2': r2_score(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions))
            }

            return metrics

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            raise RuntimeError("Failed to calculate performance metrics") from e

    def analyze_performance_trend(self, performance_history: list) -> dict[str, Any]:
        """
        Analyze performance trend over time.

        Args:
            performance_history: List of performance records with timestamps and metrics

        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Extract performance history
            recent_history = performance_history[-50:]  # Last 50 records

            if len(recent_history) < 10:
                return {'status': 'insufficient_history'}

            # Calculate trend for each metric
            trend_analysis = {
                'status': 'completed',
                'metrics_trends': {},
                'overall_degradation': False,
                'degradation_score': 0.0
            }

            # Analyze each metric
            for metric_name in ['mse', 'mae', 'r2']:
                values = [record['metrics'].get(metric_name, 0) for record in recent_history]

                if len(values) >= 10:
                    # Calculate trend using linear regression
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]

                    # Calculate degradation score
                    if metric_name in ['mse', 'mae']:  # Error metrics (higher is worse)
                        degradation_score = max(0, slope / np.mean(values)) if np.mean(values) > 0 else 0
                    else:  # R2 (lower is worse)
                        degradation_score = max(0, -slope / np.mean(values)) if np.mean(values) > 0 else 0

                    trend_analysis['metrics_trends'][metric_name] = {
                        'slope': slope,
                        'degradation_score': degradation_score,
                        'trend': 'degrading' if slope > 0.001 else 'improving' if slope < -0.001 else 'stable'
                    }

            # Calculate overall degradation score
            degradation_scores = [
                trend['degradation_score']
                for trend in trend_analysis['metrics_trends'].values()
            ]

            if degradation_scores:
                trend_analysis['degradation_score'] = np.mean(degradation_scores)
                trend_analysis['overall_degradation'] = trend_analysis['degradation_score'] > 0.05

            return trend_analysis

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error analyzing performance trend: {e}")
            return {'status': 'error', 'error': str(e)}

    def calculate_confidence_stats(self, confidences: np.ndarray) -> dict[str, float]:
        """
        Calculate confidence statistics.

        Args:
            confidences: Array of confidence values

        Returns:
            Dictionary with confidence statistics
        """
        try:
            stats_dict = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'median': np.median(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }

            return stats_dict

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating confidence stats: {e}")
            raise RuntimeError("Failed to calculate confidence statistics") from e

    def calculate_confidence_drift(self,
                                  current_confidences: np.ndarray,
                                  reference_confidences: np.ndarray) -> dict[str, Any]:
        """
        Calculate confidence drift between current and reference distributions.

        Args:
            current_confidences: Current confidence distribution
            reference_confidences: Reference confidence distribution

        Returns:
            Dictionary with confidence drift results
        """
        try:
            # Calculate current statistics
            current_stats = self.calculate_confidence_stats(current_confidences)

            # Calculate reference statistics
            ref_stats = self.calculate_confidence_stats(reference_confidences)

            # Calculate confidence drift
            mean_diff = abs(current_stats['mean'] - ref_stats['mean'])
            std_diff = abs(current_stats['std'] - ref_stats['std'])

            # Normalize differences
            mean_drift = mean_diff / ref_stats['mean'] if ref_stats['mean'] > 0 else 0
            std_drift = std_diff / ref_stats['std'] if ref_stats['std'] > 0 else 0

            # Overall drift score
            drift_score = (mean_drift + std_drift) / 2

            # Check threshold
            threshold = self.thresholds['confidence_drift']['threshold']
            drift_detected = drift_score > threshold

            result = {
                'current_stats': current_stats,
                'reference_stats': ref_stats,
                'drift_score': drift_score,
                'threshold': threshold,
                'drift_detected': drift_detected,
                'severity': self.thresholds['confidence_drift']['severity'] if drift_detected else 'none'
            }

            return result

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating confidence drift: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def calculate_overall_drift_score(self, drift_methods: dict[str, Any]) -> float:
        """
        Calculate overall drift score from multiple methods.

        Args:
            drift_methods: Dictionary of drift method results

        Returns:
            Overall drift score (0-1)
        """
        try:
            drift_scores = []
            for _method_name, method_result in drift_methods.items():
                drift_scores.append(method_result.get('drift_score', 0.0))

            if drift_scores:
                return np.mean(drift_scores)
            return 0.0

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error calculating overall drift score: {e}")
            return 0.0

    def determine_drift_severity(self, drift_score: float) -> str:
        """
        Determine drift severity based on drift score.

        Args:
            drift_score: Overall drift score (0-1)

        Returns:
            Severity level: 'critical', 'high', 'medium', 'low', 'none'
        """
        if drift_score > 0.8:
            return 'critical'
        elif drift_score > 0.6:
            return 'high'
        elif drift_score > 0.4:
            return 'medium'
        elif drift_score > 0.2:
            return 'low'
        else:
            return 'none'


# Factory function
def get_drift_calculator(thresholds: dict[str, Any] | None = None) -> DriftCalculator:
    """Factory function to get DriftCalculator instance."""
    return DriftCalculator(thresholds)
