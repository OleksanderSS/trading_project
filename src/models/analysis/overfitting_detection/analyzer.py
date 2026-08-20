from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve

from src.pipeline.stages.modeling.walk_forward_validation import (
    PurgedTimeSeriesSplit,
)

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("OverfittingAnalyzer")

class OverfittingAnalyzer:
    """Core analysis logic for overfitting detection."""

    def __init__(self, config: Any, metrics_calculator: Any):
        self.logger = logger
        self.config = config
        self.metrics = metrics_calculator

    def _splitter(self) -> PurgedTimeSeriesSplit:
        """Folds with a gap between train and validation.

        An analyzer whose entire job is to DETECT overfitting must not leak
        while measuring it. A plain TimeSeriesSplit puts validation
        immediately after training, so with a forward-looking target the last
        training labels are computed from prices inside the validation window
        -- and the resulting curve reports less overfitting than there is,
        which is the one direction this tool must never err in.
        """
        return PurgedTimeSeriesSplit(
            n_splits=self.config.cv_folds,
            purge_rows=int(getattr(self.config, 'purge_rows', 5)),
        )

    async def generate_learning_curve(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Generate learning curve data."""
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y,
                cv=self._splitter(),
                train_sizes=self.config.train_sizes,
                scoring=self.config.scoring_metric,
                n_jobs=-1
            )

            # Negate scores if using negative metrics
            if self.config.scoring_metric.startswith('neg_'):
                train_scores = -train_scores
                test_scores = -test_scores

            return {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
                'test_scores_std': np.std(test_scores, axis=1).tolist()
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error generating learning curve: {e}", exc_info=True)
            raise DataProcessingError(f"Learning curve generation failed: {e}") from e

    async def perform_cv_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Perform cross-validation analysis."""
        try:
            scores = cross_val_score(
                model, X, y, cv=self._splitter(),
                scoring=self.config.scoring_metric,
            )

            if self.config.scoring_metric.startswith('neg_'):
                scores = -scores

            return {
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'cv_folds': self.config.cv_folds
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in CV analysis: {e}", exc_info=True)
            raise DataProcessingError(f"CV analysis failed: {e}") from e

    def analyze_train_val_gap(self,
                            model: Any,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, Any]:
        """Analyze the performance gap between training and validation sets."""
        try:
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)

            train_metrics = self.metrics.calculate_metrics(y_train, train_preds)
            val_metrics = self.metrics.calculate_metrics(y_val, val_preds)

            # Calculate gap (assuming RMSE or MSE where higher is worse)
            metric_key = 'rmse' if 'rmse' in train_metrics else 'mse'
            gap = (val_metrics[metric_key] - train_metrics[metric_key]) / train_metrics[metric_key] if train_metrics[metric_key] != 0 else 0

            return {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'gap': float(gap),
                'status': 'high_gap' if gap > self.config.thresholds['train_val_gap']['threshold'] else 'normal'
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error analyzing train-val gap: {e}", exc_info=True)
            raise DataProcessingError(f"Train-val gap analysis failed: {e}") from e

    def detect_signals(self, learning_curve_res: dict[str, Any], cv_res: dict[str, Any], gap_res: dict[str, Any]) -> dict[str, Any]:
        """Detect overfitting signals based on all analysis results."""
        signals = {}
        thresholds = self.config.thresholds

        train_curve = learning_curve_res.get('train_scores_mean') or []
        test_curve = learning_curve_res.get('test_scores_mean') or []
        if train_curve and test_curve:
            final_curve_gap = abs(float(train_curve[-1]) - float(test_curve[-1]))
            curve_threshold = thresholds.get(
                'learning_curve_gap',
                thresholds['train_val_gap']
            )['threshold']
            if final_curve_gap > curve_threshold:
                signals['learning_curve_gap'] = {
                    'detected': True,
                    'value': final_curve_gap,
                    'severity': 'medium'
                }

        # 1. Train-Val Gap Signal
        if gap_res.get('gap', 0) > thresholds['train_val_gap']['threshold']:
            signals['train_val_gap'] = {
                'detected': True,
                'value': gap_res['gap'],
                'severity': 'high'
            }

        # 2. CV Variance Signal
        cv_std = cv_res.get('std', 0)
        cv_mean = cv_res.get('mean', 1)
        cv_variance_ratio = cv_std / cv_mean if cv_mean else 0.0
        if cv_variance_ratio > thresholds['cv_variance']['threshold']:
            signals['cv_variance'] = {
                'detected': True,
                'value': cv_variance_ratio,
                'severity': 'medium'
            }

        return signals

    def generate_recommendations(self, signals: dict[str, Any]) -> list[str]:
        """Generate recommendations based on detected signals."""
        recommendations = []
        if 'train_val_gap' in signals:
            recommendations.append("Increase regularization (L1/L2) to reduce the training-validation gap.")
            recommendations.append("Collect more training data or simplify the model architecture.")
        if 'cv_variance' in signals:
            recommendations.append("The model is sensitive to data splits. Consider using a more robust model or cross-validation scheme.")
        if not signals:
            recommendations.append("No significant overfitting signals detected. Model appears robust.")
        return recommendations
