#!/usr/bin/env python3
"""
Prediction Drift Monitor - Real-time Prediction Drift Monitoring
Monitors model prediction drift and triggers automatic retraining when needed.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.models.monitoring.drift.alert_system import get_alert_system
from src.models.monitoring.drift.analyzer import DriftAnalyzer
from src.models.monitoring.drift.drift_calculator import get_drift_calculator
from src.models.monitoring.drift.drift_visualizer import get_drift_visualizer
from src.models.monitoring.drift.history import HistoryManager

logger = ProjectLogger.get_logger("PredictionDriftMonitor")

class PredictionDriftMonitor:
    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = logger
        self.config = config or {}

        # Initialize components
        thresholds = self.config.get('thresholds', {})
        self.drift_calculator = get_drift_calculator(thresholds)
        self.drift_visualizer = get_drift_visualizer()
        self.alert_system = get_alert_system(self.config)

        # Monitoring settings
        self.window_size = self.config.get('window_size', 1000)
        self.reference_window_size = self.config.get('reference_window_size', 5000)
        self.min_samples_for_drift = self.config.get('min_samples_for_drift', 100)

        # History Manager and Drift Analyzer
        self.history_manager = HistoryManager(self.window_size, self.reference_window_size)
        self.drift_analyzer = DriftAnalyzer(self.drift_calculator, self.min_samples_for_drift)

        # State tracking
        self.current_drift_status = 'stable'

        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/monitoring/prediction_drift'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("✅ PredictionDriftMonitor initialized")

    async def monitor_predictions(self,
                                predictions: np.ndarray,
                                actuals: np.ndarray | None = None,
                                confidences: np.ndarray | None = None,
                                timestamp: datetime | None = None) -> dict[str, Any]:
        if timestamp is None:
            timestamp = datetime.now()

        self.logger.info(f"🔍 Monitoring {len(predictions)} predictions at {timestamp}")

        results = {
            'timestamp': timestamp,
            'predictions_count': len(predictions),
            'drift_analysis': {},
            'performance_analysis': {},
            'confidence_analysis': {},
            'retraining_recommendations': [],
            'drift_status': self.current_drift_status
        }

        try:
            # 1. Update history via HistoryManager
            self.history_manager.update_prediction_history(predictions, actuals, confidences, timestamp)

            # 2. Check data availability
            if len(self.history_manager.prediction_history) < self.min_samples_for_drift:
                results['status'] = 'insufficient_data'
                results['message'] = f'Need at least {self.min_samples_for_drift} samples, have {len(self.history_manager.prediction_history)}'
                return results

            # 3. Detect drift
            drift_analysis = await self._detect_prediction_drift(predictions, timestamp)
            results['drift_analysis'] = drift_analysis

            # 4. Performance degradation
            if actuals is not None:
                results['performance_analysis'] = self._analyze_performance_degradation(actuals, predictions, timestamp)

            # 5. Confidence drift
            if confidences is not None:
                results['confidence_analysis'] = self._analyze_confidence_drift(confidences, timestamp)

            # 6. Retraining
            results['retraining_recommendations'] = self._generate_retraining_recommendations(
                drift_analysis, results.get('performance_analysis', {}), results.get('confidence_analysis', {}), timestamp
            )

            self._update_drift_status(drift_analysis, results['retraining_recommendations'])
            results['drift_status'] = self.current_drift_status

            self._store_monitoring_results(results)
            self.logger.info(f"✅ Prediction monitoring complete. Drift status: {self.current_drift_status}")

            return results

        except Exception as e:
            self.logger.error(f"Error in prediction monitoring: {e}", exc_info=True)
            raise DataProcessingError(f"Prediction monitoring failed: {e}") from e

    async def _detect_prediction_drift(self,
                                       current_predictions: np.ndarray,
                                       timestamp: datetime) -> dict[str, Any]:
        """Detect drift in prediction distributions using DriftCalculator."""

        drift_analysis = {
            'status': 'completed',
            'drift_detected': False,
            'drift_methods': {},
            'overall_drift_score': 0.0,
            'drift_severity': 'none'
        }

        try:
            # Get reference predictions
            if len(self.reference_predictions) < self.min_samples_for_drift:
                drift_analysis['status'] = 'insufficient_reference_data'
                return drift_analysis

            reference_preds = np.array([r['prediction'] for r in self.reference_predictions])

            # Use DriftCalculator for statistical tests
            ks_result = self.drift_calculator.perform_ks_test(current_predictions, reference_preds)
            drift_analysis['drift_methods']['ks_test'] = ks_result

            psi_result = self.drift_calculator.calculate_psi(current_predictions, reference_preds)
            drift_analysis['drift_methods']['psi'] = psi_result

            wasserstein_result = self.drift_calculator.calculate_wasserstein_distance(current_predictions, reference_preds)
            drift_analysis['drift_methods']['wasserstein'] = wasserstein_result

            # Calculate overall drift score
            drift_analysis['overall_drift_score'] = self.drift_calculator.calculate_overall_drift_score(
                drift_analysis['drift_methods']
            )
            drift_analysis['drift_detected'] = drift_analysis['overall_drift_score'] > 0.2
            drift_analysis['drift_severity'] = self.drift_calculator.determine_drift_severity(
                drift_analysis['overall_drift_score']
            )

            # Store drift detection
            self.drift_history.append({
                'timestamp': timestamp,
                'drift_detected': drift_analysis['drift_detected'],
                'drift_severity': drift_analysis['drift_severity'],
                'drift_score': drift_analysis['overall_drift_score'],
                'methods': drift_analysis['drift_methods']
            })

            return drift_analysis

        except Exception as e:
            self.logger.error(f"Error detecting prediction drift: {e}")
            drift_analysis['status'] = 'error'
            drift_analysis['error'] = str(e)
            return drift_analysis

    def _analyze_performance_degradation(self,
                                      actuals: np.ndarray,
                                      predictions: np.ndarray,
                                      timestamp: datetime) -> dict[str, Any]:
        """Analyze performance degradation over time using DriftCalculator."""

        performance_analysis = {
            'status': 'completed',
            'current_performance': {},
            'performance_trend': {},
            'degradation_detected': False,
            'degradation_score': 0.0
        }

        try:
            # Calculate current performance metrics
            current_metrics = self.drift_calculator.calculate_performance_metrics(actuals, predictions)
            performance_analysis['current_performance'] = current_metrics

            # Add to performance history
            self.performance_history.append({
                'timestamp': timestamp,
                'metrics': current_metrics,
                'sample_count': len(predictions)
            })

            # Analyze performance trend if we have enough history
            if len(self.performance_history) >= 10:
                trend_analysis = self.drift_calculator.analyze_performance_trend(
                    list(self.performance_history)
                )
                performance_analysis['performance_trend'] = trend_analysis

                # Check for degradation
                degradation_score = trend_analysis.get('degradation_score', 0.0)
                performance_analysis['degradation_detected'] = degradation_score > 0.1
                performance_analysis['degradation_score'] = degradation_score

            return performance_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing performance degradation: {e}")
            performance_analysis['status'] = 'error'
            performance_analysis['error'] = str(e)
            return performance_analysis

    def _analyze_confidence_drift(self,
                                   confidences: np.ndarray,
                                   timestamp: datetime) -> dict[str, Any]:
        """Analyze confidence distribution drift using DriftCalculator."""

        confidence_analysis = {
            'status': 'completed',
            'current_confidence_stats': {},
            'confidence_drift_detected': False,
            'drift_score': 0.0
        }

        try:
            # Calculate current confidence statistics
            current_stats = self.drift_calculator.calculate_confidence_stats(confidences)
            confidence_analysis['current_confidence_stats'] = current_stats

            # Get reference confidence statistics
            reference_confidences = [
                r['confidence'] for r in self.reference_predictions
                if r['confidence'] is not None
            ]

            if len(reference_confidences) >= self.min_samples_for_drift:
                ref_confidences = np.array(reference_confidences)

                # Calculate confidence drift using DriftCalculator
                drift_result = self.drift_calculator.calculate_confidence_drift(
                    confidences, ref_confidences
                )

                confidence_analysis['confidence_drift_detected'] = drift_result['drift_detected']
                confidence_analysis['drift_score'] = drift_result['drift_score']
                confidence_analysis['reference_stats'] = drift_result['reference_stats']

            return confidence_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing confidence drift: {e}")
            confidence_analysis['status'] = 'error'
            confidence_analysis['error'] = str(e)
            return confidence_analysis

    def _generate_retraining_recommendations(self,
                                            drift_analysis: dict[str, Any],
                                            performance_analysis: dict[str, Any],
                                            confidence_analysis: dict[str, Any],
                                            timestamp: datetime) -> list[str]:
        """Generate retraining recommendations using AlertSystem."""

        return self.alert_system.generate_retraining_recommendations(
            drift_analysis, performance_analysis, confidence_analysis, timestamp
        )

    def _update_drift_status(self,
                             drift_analysis: dict[str, Any],
                             recommendations: list[str]) -> None:
        """Update current drift status based on analysis."""

        try:
            if drift_analysis.get('drift_detected', False):
                drift_severity = drift_analysis.get('drift_severity', 'low')
                self.current_drift_status = f'drift_{drift_severity}'
            else:
                self.current_drift_status = 'stable'

        except Exception as e:
            self.logger.error(f"Error updating drift status: {e}")

    def _store_monitoring_results(self, results: dict[str, Any]) -> None:
        """Store monitoring results for historical tracking."""

        try:
            # Store in JSON file
            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"prediction_drift_monitoring_{timestamp}.json"
            filepath = self.storage_path / filename

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Keep only last 100 files
            files = list(self.storage_path.glob("prediction_drift_monitoring_*.json"))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for file_to_delete in files[100:]:
                file_to_delete.unlink()

        except Exception as e:
            self.logger.error(f"Failed to store monitoring results: {e}")

    def get_drift_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of drift monitoring over time period."""

        cutoff_time = datetime.now() - timedelta(days=days)

        # Filter recent drift history
        recent_drift = [
            record for record in self.drift_history
            if record['timestamp'] >= cutoff_time
        ]

        if not recent_drift:
            return {'error': 'No recent drift monitoring data available'}

        # Analyze drift patterns
        summary = {
            'period_days': days,
            'total_drift_events': len(recent_drift),
            'drift_severity_distribution': {},
            'drift_frequency': 0.0,
            'most_common_drift_method': None,
            'drift_trend': 'stable'
        }

        # Calculate severity distribution
        severity_counts = {}
        for record in recent_drift:
            severity = record.get('drift_severity', 'none')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        summary['drift_severity_distribution'] = severity_counts

        # Calculate drift frequency
        if days > 0:
            summary['drift_frequency'] = len(recent_drift) / days

        # Get most common drift method
        method_counts = {}
        for record in recent_drift:
            methods = record.get('methods', {})
            for method_name, method_result in methods.items():
                if method_result.get('drift_detected', False):
                    method_counts[method_name] = method_counts.get(method_name, 0) + 1

        if method_counts:
            summary['most_common_drift_method'] = max(method_counts.items(), key=lambda x: x[1])[0]

        # Analyze drift trend
        if len(recent_drift) >= 5:
            recent_scores = [record.get('drift_score', 0) for record in recent_drift[-10:]]
            if len(recent_scores) >= 3:
                slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if slope > 0.01:
                    summary['drift_trend'] = 'increasing'
                elif slope < -0.01:
                    summary['drift_trend'] = 'decreasing'
                else:
                    summary['drift_trend'] = 'stable'

        return summary

    def trigger_retraining(self,
                           reason: str,
                           severity: str,
                           timestamp: datetime | None = None) -> dict[str, Any]:
        """Trigger model retraining using AlertSystem."""

        retraining_record = self.alert_system.record_retraining(reason, severity, timestamp)
        self.retraining_history.append(retraining_record)
        return retraining_record


# Factory function for easy instantiation
def get_prediction_drift_monitor(config: dict[str, Any] | None = None) -> PredictionDriftMonitor:
    """Factory function to get PredictionDriftMonitor instance."""
    return PredictionDriftMonitor(config)


# Convenience function for quick monitoring
async def monitor_predictions_quick(predictions: np.ndarray,
                                  actuals: np.ndarray | None = None,
                                  confidences: np.ndarray | None = None,
                                  config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Quick prediction drift monitoring.

    Args:
        predictions: Model predictions to monitor
        actuals: Actual values (optional)
        confidences: Prediction confidences (optional)
        config: Configuration dictionary

    Returns:
        Prediction drift monitoring result dictionary
    """
    monitor = get_prediction_drift_monitor(config)
    return await monitor.monitor_predictions(predictions, actuals, confidences)
