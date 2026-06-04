#!/usr/bin/env python3
"""
Model Analyzer - Comprehensive Model Analysis
Handles baseline, regime, overfitting, and drift analysis.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.models.analysis.baseline_dominance_detector import BaselineDominanceDetector
from src.models.analysis.overfitting_detector import OverfittingDetector
from src.models.analysis.regime_winner_analyzer import RegimeWinnerAnalyzer
from src.models.monitoring.prediction_drift_monitor import PredictionDriftMonitor

logger = ProjectLogger.get_logger("ModelAnalyzer")


class ModelAnalyzer:
    """
    Comprehensive model analyzer.

    Handles:
    - Baseline dominance analysis
    - Regime consistency analysis
    - Overfitting detection
    - Prediction drift monitoring
    - Model metrics calculation
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Model Analyzer.

        Args:
            config: Configuration dictionary for all components
        """
        self.logger = logger
        self.config = config or {}

        # Initialize all analysis components
        self.baseline_detector = BaselineDominanceDetector(self.config.get('baseline_detector', {}))
        self.regime_analyzer = RegimeWinnerAnalyzer(self.config.get('regime_analyzer', {}))
        self.overfitting_detector = OverfittingDetector(self.config.get('overfitting_detector', {}))
        self.drift_monitor = PredictionDriftMonitor(self.config.get('drift_monitor', {}))

        self.logger.info("✅ ModelAnalyzer initialized with all components")

    async def perform_baseline_analysis(self,
                                       model: Any,
                                       X_train: pd.DataFrame,
                                       y_train: pd.Series,
                                       X_val: pd.DataFrame | None,
                                       y_val: pd.Series | None) -> dict[str, Any]:
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
            return await self.baseline_detector.analyze_baseline_dominance(
                model_results, X_val if X_val is not None else X_train
            )

        except Exception as e:
            self.logger.error(f"Error in baseline analysis: {e}", exc_info=True)
            raise DataProcessingError(f"Baseline analysis failed: {e}") from e

    async def perform_regime_analysis(self,
                                      model: Any,
                                      market_data: pd.DataFrame,
                                      X_train: pd.DataFrame,
                                      y_train: pd.Series) -> dict[str, Any]:
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
            return await self.regime_analyzer.analyze_regime_consistency(
                model_results, market_data
            )

        except Exception as e:
            self.logger.error(f"Error in regime analysis: {e}", exc_info=True)
            raise DataProcessingError(f"Regime analysis failed: {e}") from e

    async def perform_overfitting_analysis(self,
                                          model: Any,
                                          X_train: pd.DataFrame,
                                          y_train: pd.Series,
                                          X_val: pd.DataFrame | None,
                                          y_val: pd.Series | None) -> dict[str, Any]:
        """Perform overfitting detection."""

        try:
            # Perform overfitting detection
            return await self.overfitting_detector.detect_overfitting(
                model, X_train, y_train, X_val, y_val
            )

        except Exception as e:
            self.logger.error(f"Error in overfitting analysis: {e}", exc_info=True)
            raise DataProcessingError(f"Overfitting analysis failed: {e}") from e

    async def perform_drift_monitoring(self,
                                       predictions: np.ndarray,
                                       actuals: np.ndarray | None,
                                       confidences: np.ndarray | None) -> dict[str, Any]:
        """Perform prediction drift monitoring."""

        try:
            # Perform drift monitoring
            return await self.drift_monitor.monitor_predictions(
                predictions, actuals, confidences
            )

        except Exception as e:
            self.logger.error(f"Error in drift monitoring: {e}", exc_info=True)
            raise DataProcessingError(f"Drift monitoring failed: {e}") from e

    def _calculate_model_metrics(self,
                                model: Any,
                                X: pd.DataFrame,
                                y: pd.Series) -> dict[str, float]:
        """Calculate model performance metrics."""

        try:
            predictions = model.predict(X)

            return {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions))
            }

        except Exception as e:
            self.logger.error(f"Error calculating model metrics: {e}", exc_info=True)
            raise DataProcessingError(f"Model metrics calculation failed: {e}") from e
