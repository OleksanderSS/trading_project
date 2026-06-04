#!/usr/bin/env python3
"""
Integrated Model Manager - Comprehensive Model Management System
Integrates all model analysis, monitoring, and management components.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.models.actions.action_trigger import ActionTrigger
from src.models.analysis.model_analyzer import ModelAnalyzer
from src.models.health.model_health_evaluator import ModelHealthEvaluator
from src.models.monitoring.prediction_drift_monitor import PredictionDriftMonitor
from src.models.registry.model_registry import ModelRegistry
from src.models.statistics.model_statistics import ModelStatistics

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

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Integrated Model Manager.

        Args:
            config: Configuration dictionary for all components
        """
        self.logger = logger
        self.config = config or {}

        # Initialize modular components
        self.model_registry = ModelRegistry(
            Path(self.config.get('storage_path', 'data/models/registry'))
        )
        self.model_analyzer = ModelAnalyzer(self.config)
        self.health_evaluator = ModelHealthEvaluator()
        self.drift_monitor = PredictionDriftMonitor(self.config.get('drift_monitor', {}))
        self.action_trigger = ActionTrigger(self.drift_monitor)
        self.model_statistics = ModelStatistics()

        # Analysis history
        self.analysis_history: list[dict[str, Any]] = []

        # Storage path for analysis results
        self.storage_path = Path(self.config.get('storage_path', 'data/models/integrated_manager'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("✅ IntegratedModelManager initialized with modular components")

    async def comprehensive_model_analysis(self,
                                       model: Any,
                                       model_name: str,
                                       X_train: pd.DataFrame,
                                       y_train: pd.Series,
                                       X_val: pd.DataFrame | None = None,
                                       y_val: pd.Series | None = None,
                                       market_data: pd.DataFrame | None = None,
                                       predictions: np.ndarray | None = None,
                                       actuals: np.ndarray | None = None,
                                       confidences: np.ndarray | None = None) -> dict[str, Any]:
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
            self.model_registry.register_model(model, model_name)

            # 1. Baseline dominance analysis
            self.logger.info("📊 Performing baseline dominance analysis...")
            baseline_results = await self.model_analyzer.perform_baseline_analysis(
                model, X_train, y_train, X_val, y_val
            )
            if isinstance(results.get('analysis_results'), dict):
                results['analysis_results']['baseline'] = baseline_results

            # 2. Regime consistency analysis (if market data available)
            if market_data is not None:
                self.logger.info("📈 Performing regime consistency analysis...")
                regime_results = await self.model_analyzer.perform_regime_analysis(
                    model, market_data, X_train, y_train
                )
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['regime'] = regime_results
            else:
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['regime'] = {'status': 'no_market_data'}

            # 3. Overfitting detection
            self.logger.info("🔍 Performing overfitting detection...")
            overfitting_results = await self.model_analyzer.perform_overfitting_analysis(
                model, X_train, y_train, X_val, y_val
            )
            if isinstance(results.get('analysis_results'), dict):
                results['analysis_results']['overfitting'] = overfitting_results

            # 4. Prediction drift monitoring (if predictions available)
            if predictions is not None:
                self.logger.info("📊 Performing prediction drift monitoring...")
                drift_results = await self.model_analyzer.perform_drift_monitoring(
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
                overall_score = self.health_evaluator.calculate_overall_health_score(analysis_results)
            else:
                overall_score = 0.5
            results['overall_health_score'] = overall_score

            # 6. Generate comprehensive recommendations
            if isinstance(analysis_results, dict):
                recommendations = self.health_evaluator.generate_comprehensive_recommendations(analysis_results, overall_score)
            else:
                recommendations = []
            results['recommendations'] = recommendations

            # 7. Determine action requirements
            results['action_required'] = self.health_evaluator.determine_action_required(recommendations)
            results['retraining_recommended'] = self.health_evaluator.determine_retraining_needed(recommendations)

            # 8. Store analysis results
            self._store_analysis_results(results)

            # 9. Trigger actions if needed
            if results['action_required']:
                await self.action_trigger.trigger_actions(results)

            self.logger.info(f"✅ Comprehensive analysis complete. Health score: {overall_score:.3f}")

            return results

        except Exception as e:
            self.logger.error(f"Error in comprehensive model analysis: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def _store_analysis_results(self, results: dict[str, Any]) -> None:
        """Store comprehensive analysis results."""

        try:
            # Update model metadata
            model_name = results['model_name']
            metadata = self.model_registry.get_model_metadata(model_name)
            if metadata:
                metadata['last_analysis'] = results['timestamp']
                metadata['analysis_count'] += 1
                self.model_registry.update_metadata(model_name, metadata)

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

    def get_model_health_summary(self, model_name: str | None = None) -> dict[str, Any]:
        """Get health summary for specific model or all models."""
        return self.model_statistics.get_model_health_summary(
            self.analysis_history,
            self.action_trigger.get_retraining_history(),
            model_name
        )


# Factory function for easy instantiation
def get_integrated_model_manager(config: dict[str, Any] | None = None) -> IntegratedModelManager:
    """Factory function to get IntegratedModelManager instance."""
    return IntegratedModelManager(config)


# Convenience function for quick comprehensive analysis
async def analyze_model_comprehensive(model: Any,
                                     model_name: str,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: pd.DataFrame | None = None,
                                     y_val: pd.Series | None = None,
                                     market_data: pd.DataFrame | None = None,
                                     predictions: np.ndarray | None = None,
                                     actuals: np.ndarray | None = None,
                                     confidences: np.ndarray | None = None,
                                     config: dict[str, Any] | None = None) -> dict[str, Any]:
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
