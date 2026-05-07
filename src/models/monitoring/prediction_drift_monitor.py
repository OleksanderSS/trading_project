#!/usr/bin/env python3
"""
Prediction Drift Monitor - Real-time Prediction Drift Monitoring
Monitors model prediction drift and triggers automatic retraining when needed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
import json
from pathlib import Path
import asyncio
from collections import defaultdict, deque
import warnings

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PredictionDriftMonitor")

class PredictionDriftMonitor:
    """
    Real-time prediction drift monitoring and automatic retraining trigger.
    
    This monitor tracks:
    - Statistical drift in prediction distributions
    - Performance degradation over time
    - Model confidence changes
    - Automatic retraining triggers
    - Historical drift patterns
    
    Critical for maintaining production model reliability.
    """
    
    # Drift detection methods
    DRIFT_METHODS = {
        'ks_test': {
            'description': 'Kolmogorov-Smirnov test for distribution drift',
            'threshold': 0.05,  # p-value threshold
            'severity': 'high'
        },
        'psi': {
            'description': 'Population Stability Index for distribution shift',
            'threshold': 0.25,  # PSI threshold
            'severity': 'medium'
        },
        'wasserstein': {
            'description': 'Wasserstein distance for distribution difference',
            'threshold': 0.1,  # Distance threshold
            'severity': 'medium'
        },
        'performance_degradation': {
            'description': 'Performance metric degradation',
            'threshold': 0.1,  # 10% degradation threshold
            'severity': 'high'
        },
        'confidence_drift': {
            'description': 'Model confidence distribution drift',
            'threshold': 0.15,  # 15% confidence change threshold
            'severity': 'medium'
        }
    }
    
    # Retraining triggers
    RETRAINING_TRIGGERS = {
        'critical_drift': {
            'description': 'Critical drift detected',
            'action': 'immediate_retraining',
            'cooldown_hours': 1
        },
        'high_drift': {
            'description': 'High drift detected',
            'action': 'scheduled_retraining',
            'cooldown_hours': 4
        },
        'medium_drift': {
            'description': 'Medium drift detected',
            'action': 'monitor_and_alert',
            'cooldown_hours': 12
        },
        'low_drift': {
            'description': 'Low drift detected',
            'action': 'log_only',
            'cooldown_hours': 24
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Prediction Drift Monitor.
        
        Args:
            config: Configuration dictionary for drift monitoring
        """
        self.logger = logger
        self.config = config or {}
        
        # Detection thresholds
        self.thresholds = self.DRIFT_METHODS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # Monitoring settings
        self.window_size = self.config.get('window_size', 1000)
        self.reference_window_size = self.config.get('reference_window_size', 5000)
        self.min_samples_for_drift = self.config.get('min_samples_for_drift', 100)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Retraining settings
        self.retraining_triggers = self.RETRAINING_TRIGGERS.copy()
        self.retraining_triggers.update(self.config.get('retraining_triggers', {}))
        
        # Data storage
        self.prediction_history = deque(maxlen=self.window_size)
        self.reference_predictions = deque(maxlen=self.reference_window_size)
        self.performance_history = deque(maxlen=1000)
        self.drift_history = []
        self.retraining_history = []
        
        # State tracking
        self.last_retraining_time = None
        self.drift_cooldowns = {}
        self.current_drift_status = 'stable'
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/monitoring/prediction_drift'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ PredictionDriftMonitor initialized")
    
    async def monitor_predictions(self, 
                                predictions: np.ndarray,
                                actuals: Optional[np.ndarray] = None,
                                confidences: Optional[np.ndarray] = None,
                                timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Monitor predictions for drift detection.
        
        Args:
            predictions: Model predictions
            actuals: Actual values (for performance monitoring)
            confidences: Prediction confidences
            timestamp: Current timestamp
            
        Returns:
            Dict with drift analysis and recommendations
        """
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
            # 1. Update prediction history
            self._update_prediction_history(predictions, actuals, confidences, timestamp)
            
            # 2. Check if we have enough data for drift detection
            if len(self.prediction_history) < self.min_samples_for_drift:
                results['status'] = 'insufficient_data'
                results['message'] = f'Need at least {self.min_samples_for_drift} samples, have {len(self.prediction_history)}'
                return results
            
            # 3. Perform drift detection
            drift_analysis = await self._detect_prediction_drift(predictions, timestamp)
            results['drift_analysis'] = drift_analysis
            
            # 4. Analyze performance degradation
            if actuals is not None:
                performance_analysis = self._analyze_performance_degradation(actuals, predictions, timestamp)
                results['performance_analysis'] = performance_analysis
            
            # 5. Analyze confidence drift
            if confidences is not None:
                confidence_analysis = self._analyze_confidence_drift(confidences, timestamp)
                results['confidence_analysis'] = confidence_analysis
            
            # 6. Generate retraining recommendations
            retraining_recommendations = self._generate_retraining_recommendations(
                drift_analysis, 
                results.get('performance_analysis', {}),
                results.get('confidence_analysis', {}),
                timestamp
            )
            results['retraining_recommendations'] = retraining_recommendations
            
            # 7. Update drift status
            self._update_drift_status(drift_analysis, retraining_recommendations)
            results['drift_status'] = self.current_drift_status
            
            # 8. Store results
            self._store_monitoring_results(results)
            
            self.logger.info(f"✅ Prediction monitoring complete. Drift status: {self.current_drift_status}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in prediction monitoring: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def _update_prediction_history(self, 
                                   predictions: np.ndarray,
                                   actuals: Optional[np.ndarray],
                                   confidences: Optional[np.ndarray],
                                   timestamp: datetime) -> None:
        """Update prediction history with new data."""
        
        try:
            # Add predictions to history
            for i, pred in enumerate(predictions):
                record = {
                    'prediction': pred,
                    'actual': actuals[i] if actuals is not None and i < len(actuals) else None,
                    'confidence': confidences[i] if confidences is not None and i < len(confidences) else None,
                    'timestamp': timestamp
                }
                self.prediction_history.append(record)
            
            # Initialize reference window if empty
            if len(self.reference_predictions) < self.reference_window_size:
                for record in list(self.prediction_history)[-len(predictions):]:
                    self.reference_predictions.append(record)
            
            self.logger.debug(f"Updated prediction history. Current size: {len(self.prediction_history)}")
            
        except Exception as e:
            self.logger.error(f"Error updating prediction history: {e}")
    
    async def _detect_prediction_drift(self, 
                                       current_predictions: np.ndarray,
                                       timestamp: datetime) -> Dict[str, Any]:
        """Detect drift in prediction distributions."""
        
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
            
            # 1. Kolmogorov-Smirnov test
            ks_result = self._perform_ks_test(current_predictions, reference_preds)
            drift_analysis['drift_methods']['ks_test'] = ks_result
            
            # 2. Population Stability Index (PSI)
            psi_result = self._calculate_psi(current_predictions, reference_preds)
            drift_analysis['drift_methods']['psi'] = psi_result
            
            # 3. Wasserstein distance
            wasserstein_result = self._calculate_wasserstein_distance(current_predictions, reference_preds)
            drift_analysis['drift_methods']['wasserstein'] = wasserstein_result
            
            # 4. Calculate overall drift score
            drift_scores = []
            for method_name, method_result in drift_analysis['drift_methods'].items():
                if method_result.get('drift_detected', False):
                    drift_scores.append(method_result.get('drift_score', 0.0))
            
            if drift_scores:
                drift_analysis['overall_drift_score'] = np.mean(drift_scores)
                drift_analysis['drift_detected'] = True
                
                # Determine severity
                if drift_analysis['overall_drift_score'] > 0.8:
                    drift_analysis['drift_severity'] = 'critical'
                elif drift_analysis['overall_drift_score'] > 0.6:
                    drift_analysis['drift_severity'] = 'high'
                elif drift_analysis['overall_drift_score'] > 0.4:
                    drift_analysis['drift_severity'] = 'medium'
                else:
                    drift_analysis['drift_severity'] = 'low'
            
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
    
    def _perform_ks_test(self, 
                           current_predictions: np.ndarray,
                           reference_predictions: np.ndarray) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        
        try:
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(current_predictions, reference_predictions)
            
            # Determine drift based on p-value
            threshold = self.thresholds['ks_test']['threshold']
            drift_detected = p_value < threshold
            
            # Calculate drift score (1 - p_value, normalized)
            drift_score = 1.0 - p_value
            
            result = {
                'method': 'ks_test',
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'threshold': threshold,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'severity': self.thresholds['ks_test']['severity'] if drift_detected else 'none'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing KS test: {e}")
            return {
                'method': 'ks_test',
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_psi(self, 
                        current_predictions: np.ndarray,
                        reference_predictions: np.ndarray,
                        bins: int = 10) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI)."""
        
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
            
            result = {
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating PSI: {e}")
            return {
                'method': 'psi',
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_wasserstein_distance(self, 
                                       current_predictions: np.ndarray,
                                       reference_predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate Wasserstein distance for drift detection."""
        
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
            
            result = {
                'method': 'wasserstein',
                'distance': distance,
                'normalized_distance': normalized_distance,
                'threshold': threshold,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'severity': self.thresholds['wasserstein']['severity'] if drift_detected else 'none'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Wasserstein distance: {e}")
            return {
                'method': 'wasserstein',
                'status': 'error',
                'error': str(e)
            }
    
    def _analyze_performance_degradation(self, 
                                      actuals: np.ndarray,
                                      predictions: np.ndarray,
                                      timestamp: datetime) -> Dict[str, Any]:
        """Analyze performance degradation over time."""
        
        performance_analysis = {
            'status': 'completed',
            'current_performance': {},
            'performance_trend': {},
            'degradation_detected': False,
            'degradation_score': 0.0
        }
        
        try:
            # Calculate current performance metrics
            current_metrics = self._calculate_performance_metrics(actuals, predictions)
            performance_analysis['current_performance'] = current_metrics
            
            # Add to performance history
            self.performance_history.append({
                'timestamp': timestamp,
                'metrics': current_metrics,
                'sample_count': len(predictions)
            })
            
            # Analyze performance trend if we have enough history
            if len(self.performance_history) >= 10:
                trend_analysis = self._analyze_performance_trend()
                performance_analysis['performance_trend'] = trend_analysis
                
                # Check for degradation
                degradation_score = trend_analysis.get('degradation_score', 0.0)
                threshold = self.thresholds['performance_degradation']['threshold']
                
                performance_analysis['degradation_detected'] = degradation_score > threshold
                performance_analysis['degradation_score'] = degradation_score
            
            return performance_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance degradation: {e}")
            performance_analysis['status'] = 'error'
            performance_analysis['error'] = str(e)
            return performance_analysis
    
    def _calculate_performance_metrics(self, 
                                     actuals: np.ndarray,
                                     predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(actuals, predictions),
                'mae': mean_absolute_error(actuals, predictions),
                'r2': r2_score(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trend over time."""
        
        try:
            # Extract performance history
            recent_history = list(self.performance_history)[-50:]  # Last 50 records
            
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
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trend: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_confidence_drift(self, 
                                   confidences: np.ndarray,
                                   timestamp: datetime) -> Dict[str, Any]:
        """Analyze confidence distribution drift."""
        
        confidence_analysis = {
            'status': 'completed',
            'current_confidence_stats': {},
            'confidence_drift_detected': False,
            'drift_score': 0.0
        }
        
        try:
            # Calculate current confidence statistics
            current_stats = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'median': np.median(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
            confidence_analysis['current_confidence_stats'] = current_stats
            
            # Get reference confidence statistics
            reference_confidences = [
                r['confidence'] for r in self.reference_predictions 
                if r['confidence'] is not None
            ]
            
            if len(reference_confidences) >= self.min_samples_for_drift:
                ref_confidences = np.array(reference_confidences)
                
                # Calculate reference statistics
                ref_stats = {
                    'mean': np.mean(ref_confidences),
                    'std': np.std(ref_confidences),
                    'median': np.median(ref_confidences),
                    'min': np.min(ref_confidences),
                    'max': np.max(ref_confidences)
                }
                
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
                confidence_analysis['confidence_drift_detected'] = drift_score > threshold
                confidence_analysis['drift_score'] = drift_score
                confidence_analysis['reference_stats'] = ref_stats
            
            return confidence_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing confidence drift: {e}")
            confidence_analysis['status'] = 'error'
            confidence_analysis['error'] = str(e)
            return confidence_analysis
    
    def _generate_retraining_recommendations(self, 
                                            drift_analysis: Dict[str, Any],
                                            performance_analysis: Dict[str, Any],
                                            confidence_analysis: Dict[str, Any],
                                            timestamp: datetime) -> List[str]:
        """Generate retraining recommendations based on analysis."""
        
        recommendations = []
        
        try:
            # Check drift analysis
            if drift_analysis.get('drift_detected', False):
                drift_severity = drift_analysis.get('drift_severity', 'low')
                
                if drift_severity == 'critical':
                    recommendations.append(
                        f"🚨 CRITICAL: Critical prediction drift detected (score: {drift_analysis.get('overall_drift_score', 0):.3f}). "
                        "Immediate retraining required."
                    )
                    recommendations.append(
                        "   → Action: Stop current model and retrain immediately."
                    )
                
                elif drift_severity == 'high':
                    recommendations.append(
                        f"⚠️ HIGH: High prediction drift detected (score: {drift_analysis.get('overall_drift_score', 0):.3f}). "
                        "Retraining recommended."
                    )
                    recommendations.append(
                        "   → Action: Schedule retraining within next 4 hours."
                    )
                
                elif drift_severity == 'medium':
                    recommendations.append(
                        f"⚠️ MEDIUM: Medium prediction drift detected (score: {drift_analysis.get('overall_drift_score', 0):.3f}). "
                        "Monitor closely."
                    )
                    recommendations.append(
                        "   → Action: Increase monitoring frequency, prepare for retraining."
                    )
                
                else:
                    recommendations.append(
                        f"📊 LOW: Low prediction drift detected (score: {drift_analysis.get('overall_drift_score', 0):.3f}). "
                        "Continue monitoring."
                    )
            
            # Check performance degradation
            if performance_analysis.get('degradation_detected', False):
                degradation_score = performance_analysis.get('degradation_score', 0.0)
                recommendations.append(
                    f"⚠️ PERFORMANCE: Performance degradation detected (score: {degradation_score:.3f}). "
                    "Retraining recommended."
                )
                
                # Add specific metric information
                metrics_trends = performance_analysis.get('performance_trend', {}).get('metrics_trends', {})
                for metric_name, trend_info in metrics_trends.items():
                    if trend_info.get('trend') == 'degrading':
                        recommendations.append(
                            f"   • {metric_name.upper()} is degrading (slope: {trend_info.get('slope', 0):.6f})"
                        )
            
            # Check confidence drift
            if confidence_analysis.get('confidence_drift_detected', False):
                drift_score = confidence_analysis.get('drift_score', 0.0)
                recommendations.append(
                    f"⚠️ CONFIDENCE: Confidence distribution drift detected (score: {drift_score:.3f}). "
                    "Model calibration may be affected."
                )
                recommendations.append(
                    "   → Action: Consider recalibration or retraining."
                )
            
            # Check cooldowns
            if self.last_retraining_time:
                hours_since_retraining = (timestamp - self.last_retraining_time).total_seconds() / 3600
                if hours_since_retraining < 24:
                    recommendations.append(
                        f"⏰ COOLDOWN: Last retraining was {hours_since_retraining:.1f} hours ago. "
                        "Consider waiting before next retraining."
                    )
            
            # No issues detected
            if not recommendations:
                recommendations.append(
                    "✅ STABLE: No significant drift or degradation detected. "
                    "Model performance is stable."
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating retraining recommendations: {e}")
            return recommendations
    
    def _update_drift_status(self, 
                             drift_analysis: Dict[str, Any],
                             recommendations: List[str]) -> None:
        """Update current drift status based on analysis."""
        
        try:
            if drift_analysis.get('drift_detected', False):
                drift_severity = drift_analysis.get('drift_severity', 'low')
                self.current_drift_status = f'drift_{drift_severity}'
            else:
                self.current_drift_status = 'stable'
            
        except Exception as e:
            self.logger.error(f"Error updating drift status: {e}")
    
    def _store_monitoring_results(self, results: Dict[str, Any]) -> None:
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
    
    def get_drift_summary(self, days: int = 30) -> Dict[str, Any]:
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
                           timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Trigger model retraining."""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        retraining_record = {
            'timestamp': timestamp,
            'reason': reason,
            'severity': severity,
            'status': 'triggered',
            'cooldown_hours': self.retraining_triggers.get(f'{severity}_drift', {}).get('cooldown_hours', 24)
        }
        
        self.retraining_history.append(retraining_record)
        self.last_retraining_time = timestamp
        
        self.logger.info(f"🔄 Retraining triggered: {reason} (severity: {severity})")
        
        return retraining_record


# Factory function for easy instantiation
def get_prediction_drift_monitor(config: Optional[Dict[str, Any]] = None) -> PredictionDriftMonitor:
    """Factory function to get PredictionDriftMonitor instance."""
    return PredictionDriftMonitor(config)


# Convenience function for quick monitoring
async def monitor_predictions_quick(predictions: np.ndarray,
                                  actuals: Optional[np.ndarray] = None,
                                  confidences: Optional[np.ndarray] = None,
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
