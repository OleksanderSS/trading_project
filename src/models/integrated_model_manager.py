#!/usr/bin/env python3
"""
Integrated Model Manager - Comprehensive Model Management System
Integrates all model analysis, monitoring, and management components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger
from src.models.analysis.baseline_dominance_detector import BaselineDominanceDetector
from src.models.analysis.regime_winner_analyzer import RegimeWinnerAnalyzer
from src.models.analysis.overfitting_detector import OverfittingDetector
from src.models.monitoring.prediction_drift_monitor import PredictionDriftMonitor

logger = ProjectLogger.get_logger("IntegratedModelManager")

class IntegratedModelManager:
    """
    Comprehensive model management system that integrates all analysis and monitoring components.
    
    This manager provides:
    - Baseline dominance detection for over-engineering prevention
    - Regime-specific model consistency analysis
    - Advanced overfitting detection and prevention
    - Real-time prediction drift monitoring
    - Comprehensive model health reporting
    - Automatic retraining triggers
    
    Critical for maintaining robust and reliable model performance in production.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Integrated Model Manager.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.logger = logger
        self.config = config or {}
        
        # Initialize all components
        self.baseline_detector = BaselineDominanceDetector(self.config.get('baseline_detector', {}))
        self.regime_analyzer = RegimeWinnerAnalyzer(self.config.get('regime_analyzer', {}))
        self.overfitting_detector = OverfittingDetector(self.config.get('overfitting_detector', {}))
        self.drift_monitor = PredictionDriftMonitor(self.config.get('drift_monitor', {}))
        
        # Model registry
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        
        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        self.retraining_history: List[Dict[str, Any]] = []
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/models/integrated_manager'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ IntegratedModelManager initialized with all components")
    
    async def comprehensive_model_analysis(self, 
                                       model: Any,
                                       model_name: str,
                                       X_train: pd.DataFrame,
                                       y_train: pd.Series,
                                       X_val: Optional[pd.DataFrame] = None,
                                       y_val: Optional[pd.Series] = None,
                                       market_data: Optional[pd.DataFrame] = None,
                                       predictions: Optional[np.ndarray] = None,
                                       actuals: Optional[np.ndarray] = None,
                                       confidences: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model analysis using all components.
        
        Args:
            model: Trained model to analyze
            model_name: Name of the model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            market_data: Market data for regime analysis (optional)
            predictions: Current predictions for drift monitoring (optional)
            actuals: Actual values for performance monitoring (optional)
            confidences: Prediction confidences (optional)
            
        Returns:
            Dict with comprehensive analysis results and recommendations
        """
        self.logger.info(f"🔍 Starting comprehensive analysis for model: {model_name}")
        
        results = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'model_type': type(model).__name__,
            'analysis_results': {},
            'overall_health_score': 0.0,
            'recommendations': [],
            'action_required': False,
            'retraining_recommended': False
        }
        
        try:
            # Register model
            self._register_model(model, model_name)
            
            # 1. Baseline dominance analysis
            self.logger.info("📊 Performing baseline dominance analysis...")
            baseline_results = await self._perform_baseline_analysis(
                model, X_train, y_train, X_val, y_val
            )
            if isinstance(results.get('analysis_results'), dict):
                results['analysis_results']['baseline'] = baseline_results
            
            # 2. Regime consistency analysis (if market data available)
            if market_data is not None:
                self.logger.info("📈 Performing regime consistency analysis...")
                regime_results = await self._perform_regime_analysis(
                    model, market_data, X_train, y_train
                )
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['regime'] = regime_results
            else:
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['regime'] = {'status': 'no_market_data'}
            
            # 3. Overfitting detection
            self.logger.info("🔍 Performing overfitting detection...")
            overfitting_results = await self._perform_overfitting_analysis(
                model, X_train, y_train, X_val, y_val
            )
            if isinstance(results.get('analysis_results'), dict):
                results['analysis_results']['overfitting'] = overfitting_results
            
            # 4. Prediction drift monitoring (if predictions available)
            if predictions is not None:
                self.logger.info("📊 Performing prediction drift monitoring...")
                drift_results = await self._perform_drift_monitoring(
                    predictions, actuals, confidences
                )
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['drift'] = drift_results
            else:
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['drift'] = {'status': 'no_predictions'}
            
            # 5. Calculate overall health score
            analysis_results = results.get('analysis_results', {})
            if isinstance(analysis_results, dict):
                overall_score = self._calculate_overall_health_score(analysis_results)
            else:
                overall_score = 0.5
            results['overall_health_score'] = overall_score
            
            # 6. Generate comprehensive recommendations
            if isinstance(analysis_results, dict):
                recommendations = self._generate_comprehensive_recommendations(analysis_results, overall_score)
            else:
                recommendations = []
            results['recommendations'] = recommendations
            
            # 7. Determine action requirements
            results['action_required'] = self._determine_action_required(recommendations)
            results['retraining_recommended'] = self._determine_retraining_needed(recommendations)
            
            # 8. Store analysis results
            self._store_analysis_results(results)
            
            # 9. Trigger actions if needed
            if results['action_required']:
                await self._trigger_actions(results)
            
            self.logger.info(f"✅ Comprehensive analysis complete. Health score: {health_score:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive model analysis: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    async def _perform_baseline_analysis(self, 
                                       model: Any,
                                       X_train: pd.DataFrame,
                                       y_train: pd.Series,
                                       X_val: Optional[pd.DataFrame],
                                       y_val: Optional[pd.Series]) -> Dict[str, Any]:
        """Perform baseline dominance analysis."""
        
        try:
            # Create model results dictionary for baseline detector
            model_results = {
                'complex_model': {
                    'model': model,
                    'predictions': model.predict(X_val) if X_val is not None else model.predict(X_train),
                    'metrics': self._calculate_model_metrics(model, X_val, y_val) if X_val is not None else self._calculate_model_metrics(model, X_train, y_train)
                }
            }
            
            # Perform baseline analysis
            baseline_result = await self.baseline_detector.analyze_baseline_dominance(
                model_results, X_val if X_val is not None else X_train
            )
            
            return baseline_result
            
        except Exception as e:
            self.logger.error(f"Error in baseline analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _perform_regime_analysis(self, 
                                      model: Any,
                                      market_data: pd.DataFrame,
                                      X_train: pd.DataFrame,
                                      y_train: pd.Series) -> Dict[str, Any]:
        """Perform regime consistency analysis."""
        
        try:
            # Create model results for regime analyzer
            model_results = {
                type(model).__name__: {
                    'model': model,
                    'predictions': model.predict(X_train),
                    'metrics': self._calculate_model_metrics(model, X_train, y_train),
                    'model_type': type(model).__name__
                }
            }
            
            # Perform regime analysis
            regime_result = await self.regime_analyzer.analyze_regime_consistency(
                model_results, market_data
            )
            
            return regime_result
            
        except Exception as e:
            self.logger.error(f"Error in regime analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _perform_overfitting_analysis(self, 
                                          model: Any,
                                          X_train: pd.DataFrame,
                                          y_train: pd.Series,
                                          X_val: Optional[pd.DataFrame],
                                          y_val: Optional[pd.Series]) -> Dict[str, Any]:
        """Perform overfitting detection."""
        
        try:
            # Perform overfitting detection
            overfitting_result = await self.overfitting_detector.detect_overfitting(
                model, X_train, y_train, X_val, y_val
            )
            
            return overfitting_result
            
        except Exception as e:
            self.logger.error(f"Error in overfitting analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _perform_drift_monitoring(self, 
                                       predictions: np.ndarray,
                                       actuals: Optional[np.ndarray],
                                       confidences: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform prediction drift monitoring."""
        
        try:
            # Perform drift monitoring
            drift_result = await self.drift_monitor.monitor_predictions(
                predictions, actuals, confidences
            )
            
            return drift_result
            
        except Exception as e:
            self.logger.error(f"Error in drift monitoring: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_model_metrics(self, 
                                model: Any,
                                X: pd.DataFrame,
                                y: pd.Series) -> Dict[str, float]:
        """Calculate model performance metrics."""
        
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            predictions = model.predict(X)
            
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating model metrics: {e}")
            return {}
    
    def _calculate_overall_health_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall model health score from all analysis results."""
        
        try:
            health_scores = []
            
            # Baseline analysis score
            baseline_result = analysis_results.get('baseline', {})
            if baseline_result.get('status') == 'completed':
                # Score based on whether complex model dominates baselines
                if not baseline_result.get('baseline_dominance_detected', True):
                    health_scores.append(0.8)  # Good - no baseline dominance
                else:
                    health_scores.append(0.3)  # Poor - baseline dominates
            
            # Regime analysis score
            regime_result = analysis_results.get('regime', {})
            if regime_result.get('status') == 'completed':
                consistency = regime_result.get('consistency_analysis', {}).get('overall_consistency', 0.5)
                health_scores.append(consistency)
            
            # Overfitting analysis score
            overfitting_result = analysis_results.get('overfitting', {})
            if overfitting_result.get('status') == 'completed':
                signal_count = overfitting_result.get('overfitting_signals', {}).get('total_signals', 0)
                # Inverse relationship - fewer signals = better health
                overfitting_score = max(0.0, 1.0 - (signal_count * 0.2))
                health_scores.append(overfitting_score)
            
            # Drift analysis score
            drift_result = analysis_results.get('drift', {})
            if drift_result.get('status') == 'completed':
                drift_status = drift_result.get('drift_status', 'stable')
                if drift_status == 'stable':
                    health_scores.append(0.9)
                elif 'low' in drift_status:
                    health_scores.append(0.7)
                elif 'medium' in drift_status:
                    health_scores.append(0.5)
                elif 'high' in drift_status:
                    health_scores.append(0.3)
                elif 'critical' in drift_status:
                    health_scores.append(0.1)
                else:
                    health_scores.append(0.5)
            
            # Calculate overall score
            if health_scores:
                return float(np.mean(health_scores))
            else:
                return 0.5  # Default score if no analysis completed
            
        except Exception as e:
            self.logger.error(f"Error calculating overall health score: {e}")
            return 0.5
    
    def _generate_comprehensive_recommendations(self, 
                                            analysis_results: Dict[str, Any],
                                            health_score: float) -> List[str]:
        """Generate comprehensive recommendations from all analysis results."""
        
        recommendations = []
        
        try:
            # Overall health recommendation
            if health_score >= 0.8:
                recommendations.append(
                    f"✅ EXCELLENT: Model health score is {health_score:.3f}. "
                    "Model is performing well."
                )
            elif health_score >= 0.6:
                recommendations.append(
                    f"⚠️ GOOD: Model health score is {health_score:.3f}. "
                    "Model is performing adequately but monitor closely."
                )
            elif health_score >= 0.4:
                recommendations.append(
                    f"⚠️ FAIR: Model health score is {health_score:.3f}. "
                    "Model has issues that need attention."
                )
            else:
                recommendations.append(
                    f"🚨 POOR: Model health score is {health_score:.3f}. "
                    "Model has significant issues requiring immediate action."
                )
            
            # Baseline analysis recommendations
            baseline_result = analysis_results.get('baseline', {})
            if baseline_result.get('baseline_dominance_detected', False):
                recommendations.append(
                    "🔧 BASELINE: Simple baseline models outperform complex model. "
                    "Consider simplifying the model architecture."
                )
            
            # Regime analysis recommendations
            regime_result = analysis_results.get('regime', {})
            if regime_result.get('status') == 'completed':
                regime_recommendations = regime_result.get('recommendations', [])
                recommendations.extend(regime_recommendations)
            
            # Overfitting analysis recommendations
            overfitting_result = analysis_results.get('overfitting', {})
            if overfitting_result.get('status') == 'completed':
                overfitting_recommendations = overfitting_result.get('recommendations', [])
                recommendations.extend(overfitting_recommendations)
            
            # Drift analysis recommendations
            drift_result = analysis_results.get('drift', {})
            if drift_result.get('status') == 'completed':
                drift_recommendations = drift_result.get('retraining_recommendations', [])
                recommendations.extend(drift_recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive recommendations: {e}")
            return recommendations
    
    def _determine_action_required(self, recommendations: List[str]) -> bool:
        """Determine if immediate action is required based on recommendations."""
        
        try:
            # Check for critical indicators
            critical_keywords = [
                'CRITICAL', 'IMMEDIATE', 'STOP', 'DANGER', 'HIGH RISK'
            ]
            
            for recommendation in recommendations:
                if any(keyword in recommendation.upper() for keyword in critical_keywords):
                    return True
            
            # Check for high severity issues
            high_severity_keywords = [
                'HIGH', 'RETRAINING REQUIRED', 'OVERFITTING', 'DRIFT DETECTED'
            ]
            
            high_severity_count = sum(
                1 for recommendation in recommendations
                if any(keyword in recommendation.upper() for keyword in high_severity_keywords)
            )
            
            return high_severity_count >= 2
            
        except Exception as e:
            self.logger.error(f"Error determining action required: {e}")
            return False
    
    def _determine_retraining_needed(self, recommendations: List[str]) -> bool:
        """Determine if retraining is recommended based on recommendations."""
        
        try:
            retraining_keywords = [
                'RETRAIN', 'RETRAINING', 'DEGRADATION', 'DRIFT', 'OVERFITTING'
            ]
            
            return any(
                keyword in recommendation.upper()
                for recommendation in recommendations
                for keyword in retraining_keywords
            )
            
        except Exception as e:
            self.logger.error(f"Error determining retraining needed: {e}")
            return False
    
    def _register_model(self, model: Any, model_name: str) -> None:
        """Register model in the model registry."""
        
        try:
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'registered_at': datetime.now(),
                'model_type': type(model).__name__,
                'last_analysis': None,
                'analysis_count': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
    
    def _store_analysis_results(self, results: Dict[str, Any]) -> None:
        """Store comprehensive analysis results."""
        
        try:
            # Update model metadata
            model_name = results['model_name']
            if model_name in self.model_metadata:
                self.model_metadata[model_name]['last_analysis'] = results['timestamp']
                self.model_metadata[model_name]['analysis_count'] += 1
            
            # Store in analysis history
            self.analysis_history.append(results)
            
            # Keep only last 1000 analyses
            if len(self.analysis_history) > 1000:
                self.analysis_history = self.analysis_history[-1000:]
            
            # Store in JSON file
            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_analysis_{model_name}_{timestamp}.json"
            filepath = self.storage_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Keep only last 100 files per model
            model_files = list(self.storage_path.glob(f"comprehensive_analysis_{model_name}_*.json"))
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for file_to_delete in model_files[100:]:
                file_to_delete.unlink()
                
        except Exception as e:
            self.logger.error(f"Failed to store analysis results: {e}")
    
    async def _trigger_actions(self, results: Dict[str, Any]) -> None:
        """Trigger required actions based on analysis results."""
        
        try:
            model_name = results['model_name']
            
            # Trigger retraining if needed
            if results['retraining_recommended']:
                retraining_reason = f"Comprehensive analysis indicated retraining needed for {model_name}"
                severity = 'high' if results['overall_health_score'] < 0.4 else 'medium'
                
                retraining_record = self.drift_monitor.trigger_retraining(
                    reason=retraining_reason,
                    severity=severity
                )
                
                self.retraining_history.append(retraining_record)
                self.logger.info(f"🔄 Retraining triggered for {model_name}: {retraining_reason}")
            
            # Send alerts for critical issues
            if results['action_required']:
                await self._send_critical_alert(results)
            
        except Exception as e:
            self.logger.error(f"Error triggering actions: {e}")
    
    async def _send_critical_alert(self, results: Dict[str, Any]) -> None:
        """Send critical alert for model issues."""
        
        try:
            model_name = results['model_name']
            health_score = results['overall_health_score']
            
            alert_message = (
                f"🚨 CRITICAL MODEL ALERT 🚨\n"
                f"Model: {model_name}\n"
                f"Health Score: {health_score:.3f}\n"
                f"Time: {results['timestamp']}\n"
                f"Recommendations:\n"
            )
            
            for i, recommendation in enumerate(results['recommendations'][:5], 1):
                alert_message += f"{i}. {recommendation}\n"
            
            self.logger.critical(alert_message)
            
            # Here you could integrate with alert systems like:
            # - Email notifications
            # - Slack notifications
            # - PagerDuty alerts
            # - SMS alerts
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert: {e}")
    
    def get_model_health_summary(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health summary for specific model or all models."""
        
        try:
            if model_name:
                # Get specific model analysis
                model_analyses = [
                    analysis for analysis in self.analysis_history
                    if analysis['model_name'] == model_name
                ]
            else:
                # Get all model analyses
                model_analyses = self.analysis_history
            
            if not model_analyses:
                return {'error': 'No analysis data available'}
            
            # Calculate summary statistics
            summary = {
                'model_name': model_name or 'all_models',
                'total_analyses': len(model_analyses),
                'latest_health_score': model_analyses[-1]['overall_health_score'],
                'health_trend': self._calculate_health_trend(model_analyses),
                'common_issues': self._get_common_issues(model_analyses),
                'retraining_frequency': self._calculate_retraining_frequency(model_name)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting model health summary: {e}")
            return {'error': str(e)}
    
    def _calculate_health_trend(self, analyses: List[Dict[str, Any]]) -> str:
        """Calculate health score trend."""
        
        try:
            if len(analyses) < 5:
                return 'insufficient_data'
            
            # Get recent health scores
            recent_scores = [analysis['overall_health_score'] for analysis in analyses[-10:]]
            
            # Calculate trend
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'degrading'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Error calculating health trend: {e}")
            return 'unknown'
    
    def _get_common_issues(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Get most common issues from analyses."""
        
        try:
            issue_counts: Dict[str, int] = {}
            
            for analysis in analyses:
                if isinstance(analysis, dict) and 'recommendations' in analysis:
                    recommendations = analysis['recommendations']
                else:
                    recommendations = []
                
                for recommendation in recommendations:
                    # Extract issue type from recommendation
                    if 'BASELINE' in recommendation.upper():
                        issue_counts['baseline_dominance'] = issue_counts.get('baseline_dominance', 0) + 1
                    elif 'OVERFITTING' in recommendation.upper():
                        issue_counts['overfitting'] = issue_counts.get('overfitting', 0) + 1
                    elif 'DRIFT' in recommendation.upper():
                        issue_counts['drift'] = issue_counts.get('drift', 0) + 1
                    elif 'REGIME' in recommendation.upper():
                        issue_counts['regime_inconsistency'] = issue_counts.get('regime_inconsistency', 0) + 1
            
            # Return top issues
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            return [issue[0] for issue in sorted_issues[:5]]
            
        except Exception as e:
            self.logger.error(f"Error getting common issues: {e}")
            return []
    
    def _calculate_retraining_frequency(self, model_name: Optional[str]) -> float:
        """Calculate retraining frequency for model."""
        
        try:
            if model_name:
                model_retrainings = [
                    record for record in self.retraining_history
                    if model_name in record.get('reason', '')
                ]
            else:
                model_retrainings = self.retraining_history
            
            if not model_retrainings:
                return 0.0
            
            # Calculate frequency over last 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            recent_retrainings = [
                record for record in model_retrainings
                if record['timestamp'] >= cutoff_time
            ]
            
            return len(recent_retrainings) / 30.0  # Retrainings per day
            
        except Exception as e:
            self.logger.error(f"Error calculating retraining frequency: {e}")
            return 0.0


# Factory function for easy instantiation
def get_integrated_model_manager(config: Optional[Dict[str, Any]] = None) -> IntegratedModelManager:
    """Factory function to get IntegratedModelManager instance."""
    return IntegratedModelManager(config)


# Convenience function for quick comprehensive analysis
async def analyze_model_comprehensive(model: Any,
                                     model_name: str,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: Optional[pd.DataFrame] = None,
                                     y_val: Optional[pd.Series] = None,
                                     market_data: Optional[pd.DataFrame] = None,
                                     predictions: Optional[np.ndarray] = None,
                                     actuals: Optional[np.ndarray] = None,
                                     confidences: Optional[np.ndarray] = None,
                                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick comprehensive model analysis.
    
    Args:
        model: Model to analyze
        model_name: Name of the model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        market_data: Market data for regime analysis (optional)
        predictions: Current predictions (optional)
        actuals: Actual values (optional)
        confidences: Prediction confidences (optional)
        config: Configuration dictionary
        
    Returns:
        Comprehensive analysis result dictionary
    """
    manager = get_integrated_model_manager(config)
    return await manager.comprehensive_model_analysis(
        model, model_name, X_train, y_train, X_val, y_val,
        market_data, predictions, actuals, confidences
    )
