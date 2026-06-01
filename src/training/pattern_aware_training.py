#!/usr/bin/env python3
"""
Pattern-aware model training.

This lightweight trainer uses market-pattern hints to choose conservative
hyperparameters, trains a small model cohort, and returns a champion.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class PatternAwareModelTrainer:
    """Train simple supervised models with market-condition-aware defaults."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._get_default_config()
        self.model_registry = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
        }
        self.training_history: list[dict[str, Any]] = []
        self.model_performance: dict[str, dict[str, Any]] = {}

    def train_pattern_aware_models(
        self,
        features: Any = None,
        targets: Any = None,
        patterns: dict[str, Any] | None = None,
        model_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train configured models and select the best validation performer."""
        if features is None or targets is None:
            return {'status': 'skipped', 'reason': 'missing_training_data'}

        patterns = patterns or {}
        conditions = self._analyze_market_conditions(features, patterns)
        training_data = self._prepare_training_data(features, targets, conditions)
        if training_data['X'].empty or training_data['y'].empty:
            return {'status': 'skipped', 'reason': 'empty_training_data'}

        selected_models = model_names or self.config.get('model_names', [
            'linear',
            'ridge',
            'random_forest',
        ])
        trained_models = {}
        for model_name in selected_models:
            if model_name not in self.model_registry:
                logger.warning(f"Unknown pattern-aware model '{model_name}', skipping.")
                continue
            params = self._get_adaptive_parameters(model_name, conditions)
            trained_models[model_name] = self._train_single_model(
                model_name,
                training_data,
                params,
                conditions,
            )

        results = self._analyze_training_results(trained_models, conditions)
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'conditions': conditions,
            'results': results,
        })
        return results

    def _analyze_market_conditions(self, features: Any, patterns: dict[str, Any]) -> dict[str, Any]:
        """Summarize market conditions used for adaptive model parameters."""
        volatility = self._estimate_volatility(features)
        data_quality = self._estimate_data_quality(features)
        regime = patterns.get('regime') or patterns.get('market_regime')
        if not regime:
            if volatility >= 0.35:
                regime = 'volatile'
            elif volatility <= 0.10:
                regime = 'calm'
            else:
                regime = 'normal'

        return {
            'regime': str(regime),
            'volatility': float(volatility),
            'data_quality': float(data_quality),
            'pattern_count': len(patterns),
        }

    def _estimate_volatility(self, price_features: Any) -> float:
        """Estimate annualized volatility from close prices or numeric feature changes."""
        frame = self._create_feature_matrix(price_features)
        if frame.empty:
            return 0.0

        if 'close' in frame.columns:
            returns = frame['close'].astype(float).pct_change(fill_method=None).dropna()
        else:
            returns = frame.select_dtypes(include=[np.number]).pct_change(fill_method=None).stack().replace([np.inf, -np.inf], np.nan).dropna()

        if returns.empty:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    def _estimate_data_quality(self, quality_features: Any) -> float:
        """Estimate usable-data ratio across numeric features."""
        frame = self._create_feature_matrix(quality_features)
        if frame.empty:
            return 0.0
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            return 0.0
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        return float(finite.mean())

    def _prepare_training_data(self, features: Any, targets: Any, conditions: dict[str, Any]) -> dict[str, Any]:
        """Build aligned feature and target arrays for time-aware validation."""
        X = self._create_feature_matrix(features)
        y = self._create_target_vector(targets)
        rows = min(len(X), len(y))
        X = X.iloc[:rows].reset_index(drop=True)
        y = y.iloc[:rows].reset_index(drop=True)

        if conditions.get('data_quality', 0.0) < 0.5:
            X = X.dropna(axis=1, thresh=max(1, int(len(X) * 0.5)))

        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return {'X': X, 'y': y}

    def _create_feature_matrix(self, features: Any) -> pd.DataFrame:
        """Construct a numeric feature matrix from DataFrame, Series, or dict input."""
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, pd.Series):
            frame = features.to_frame()
        elif isinstance(features, dict):
            normalized = {}
            for key, value in features.items():
                if np.isscalar(value):
                    normalized[key] = [value]
                else:
                    normalized[key] = value
            frame = pd.DataFrame(normalized)
        else:
            frame = pd.DataFrame(features)

        drop_cols = [col for col in frame.columns if str(col).startswith('target_')]
        return frame.drop(columns=drop_cols, errors='ignore')

    def _create_target_vector(self, targets: Any) -> pd.Series:
        """Construct a supervised target vector."""
        if isinstance(targets, pd.Series):
            series = targets.copy()
        elif isinstance(targets, pd.DataFrame):
            target_cols = [col for col in targets.columns if str(col).startswith('target_')]
            selected_col = target_cols[0] if target_cols else targets.select_dtypes(include=[np.number]).columns[0]
            series = targets[selected_col].copy()
        elif isinstance(targets, dict):
            frame = pd.DataFrame(targets)
            target_cols = [col for col in frame.columns if str(col).startswith('target_')]
            selected_col = target_cols[0] if target_cols else frame.columns[0]
            series = frame[selected_col].copy()
        else:
            series = pd.Series(targets)
        return series.astype(float)

    def _get_adaptive_parameters(self, model_name: str, conditions: dict[str, Any]) -> dict[str, Any]:
        """Select conservative hyperparameters for the detected regime."""
        regime = conditions.get('regime', 'normal')
        if model_name == 'random_forest':
            return {
                'n_estimators': 80 if regime == 'volatile' else 120,
                'max_depth': 4 if regime == 'volatile' else 6,
                'min_samples_leaf': 3 if regime == 'volatile' else 1,
                'random_state': 42,
            }
        if model_name == 'gradient_boosting':
            return {
                'n_estimators': 80,
                'learning_rate': 0.03 if regime == 'volatile' else 0.05,
                'max_depth': 2 if regime == 'volatile' else 3,
                'random_state': 42,
            }
        if model_name in {'ridge', 'lasso'}:
            return {'alpha': 2.0 if regime == 'volatile' else 1.0}
        return {}

    def _train_single_model(
        self,
        model_name: str,
        training_data: dict[str, Any],
        params: dict[str, Any],
        conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """Train one model with a chronological validation split."""
        X = training_data['X']
        y = training_data['y']
        split_idx = max(1, int(len(X) * 0.8))
        if split_idx >= len(X):
            split_idx = max(1, len(X) - 1)

        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        if X_val.empty:
            X_val, y_val = X_train, y_train

        model = self.model_registry[model_name](**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        metrics = {
            'mse': float(mean_squared_error(y_val, predictions)),
            'mae': float(mean_absolute_error(y_val, predictions)),
            'r2': float(r2_score(y_val, predictions)) if len(y_val) > 1 else 0.0,
            'validation_rows': int(len(X_val)),
        }
        result = {
            'model_name': model_name,
            'model': model,
            'params': params,
            'metrics': metrics,
            'conditions': conditions,
        }
        self.model_performance[model_name] = metrics
        return result

    def _analyze_training_results(self, trained_models: dict[str, Any], conditions: dict[str, Any]) -> dict[str, Any]:
        """Aggregate model metrics and choose a champion."""
        if not trained_models:
            return {'status': 'failed', 'reason': 'no_models_trained', 'conditions': conditions}

        best_model = self._select_best_model(trained_models, {})
        model_metrics = {
            name: payload['metrics']
            for name, payload in trained_models.items()
        }
        return {
            'status': 'success',
            'conditions': conditions,
            'models_trained': list(trained_models.keys()),
            'model_metrics': model_metrics,
            'best_model': best_model,
        }

    def _select_best_model(self, trained_models: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
        """Select the best model by R2 first, then lower MSE."""
        del results
        best_name, best_payload = max(
            trained_models.items(),
            key=lambda item: (item[1]['metrics']['r2'], -item[1]['metrics']['mse']),
        )
        return {
            'model_name': best_name,
            'metrics': best_payload['metrics'],
            'params': best_payload['params'],
        }

    def _get_default_config(self) -> dict[str, Any]:
        """Default pattern-aware training configuration."""
        return {
            'model_names': ['linear', 'ridge', 'random_forest'],
        }


def train_pattern_aware_models(
    features: Any = None,
    targets: Any = None,
    patterns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience entry point for pattern-aware model training."""
    trainer = PatternAwareModelTrainer()
    return trainer.train_pattern_aware_models(features, targets, patterns)
