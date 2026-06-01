import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.model_selection import learning_curve, cross_val_score, TimeSeriesSplit
import logging
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger("OverfittingAnalyzer")

class OverfittingAnalyzer:
    """Core analysis logic for overfitting detection."""
    
    def __init__(self, config: Any, metrics_calculator: Any):
        self.logger = logger
        self.config = config
        self.metrics = metrics_calculator

    async def generate_learning_curve(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate learning curve data."""
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, 
                cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
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
        except Exception as e:
            self.logger.error(f"Error generating learning curve: {e}", exc_info=True)
            raise DataProcessingError(f"Learning curve generation failed: {e}") from e

    async def perform_cv_analysis(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation analysis."""
        try:
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=self.config.cv_folds), scoring=self.config.scoring_metric)
            
            if self.config.scoring_metric.startswith('neg_'):
                scores = -scores
                
            return {
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'cv_folds': self.config.cv_folds
            }
        except Exception as e:
            self.logger.error(f"Error in CV analysis: {e}", exc_info=True)
            raise DataProcessingError(f"CV analysis failed: {e}") from e

    def analyze_train_val_gap(self, 
                            model: Any, 
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
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
        except Exception as e:
            self.logger.error(f"Error analyzing train-val gap: {e}", exc_info=True)
            raise DataProcessingError(f"Train-val gap analysis failed: {e}") from e

    def detect_signals(self, learning_curve_res: Dict[str, Any], cv_res: Dict[str, Any], gap_res: Dict[str, Any]) -> Dict[str, Any]:
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

    def generate_recommendations(self, signals: Dict[str, Any]) -> List[str]:
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
