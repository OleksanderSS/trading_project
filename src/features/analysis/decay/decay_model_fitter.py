#!/usr/bin/env python3
"""
Decay Model Fitter - Model Fitting Logic
Handles decay function fitting and optimization.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DecayModelFitter")


class DecayModelFitter:
    """
    Decay model fitter for fitting decay functions to data.

    Handles:
    - Decay function fitting
    - Parameter optimization
    - Model performance evaluation
    - Model selection
    """

    def __init__(self, optimization_enabled: bool = True):
        """
        Initialize Decay Model Fitter.

        Args:
            optimization_enabled: Whether to enable parameter optimization
        """
        self.logger = logger
        self.optimization_enabled = optimization_enabled
        self.logger.info("✅ DecayModelFitter initialized")

    def fit_decay_function(self,
                          news_data: pd.DataFrame,
                          function_name: str,
                          function_config: dict[str, Any],
                          typical_hours: float) -> dict[str, Any]:
        """Fit a specific decay function to news data."""
        try:
            X = news_data[['hours_since_news']].values
            y = news_data['returns'].values

            def decay_function_with_params(params, hours):
                return function_config['function'](hours, *params)

            initial_params = self._get_initial_parameters(function_name, typical_hours)

            def objective(params):
                try:
                    predicted_impact = decay_function_with_params(params, X.flatten())
                    mse = mean_squared_error(y, predicted_impact)
                    return mse
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Error in objective function: {e}', exc_info=True)
                    return float('inf')

            if self.optimization_enabled:
                result = minimize(objective, initial_params, method='L-BFGS', options={'maxiter': 100})
                if result.success:
                    optimal_params = result.x
                    optimization_success = True
                else:
                    optimal_params = initial_params
                    optimization_success = False
            else:
                optimal_params = initial_params
                optimization_success = True

            predicted_impact = decay_function_with_params(optimal_params, X.flatten())
            performance = {
                'mse': mean_squared_error(y, predicted_impact),
                'mae': mean_absolute_error(y, predicted_impact),
                'r2': self._calculate_r2(y, predicted_impact),
                'optimization_success': optimization_success
            }

            model_info = {
                'function_name': function_name,
                'parameters': dict(zip(function_config['params'], optimal_params, strict=False)),
                'performance': performance,
                'description': function_config['description']
            }

            return model_info
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error fitting decay function {function_name}: {e}', exc_info=True)
            return {'function_name': function_name, 'error': str(e)}

    def _get_initial_parameters(self, function_name: str, typical_hours: float) -> list[float]:
        """Get initial parameter guess for optimization."""
        if function_name == 'exponential':
            return [typical_hours, 1.0]
        elif function_name == 'linear':
            return [typical_hours * 2, 1.0]
        elif function_name == 'logarithmic':
            return [typical_hours, 1.0]
        elif function_name == 'step_function':
            return [1.0, 0.1, 1.0]
        elif function_name == 'power_law':
            return [1.0, typical_hours, 1.0]
        else:
            return [1.0, 1.0]

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0
        r2 = 1 - ss_res / ss_tot
        return float(r2)

    def select_best_overall_model(self, models: dict[str, Any]) -> dict[str, Any]:
        """Select best overall model across all news types."""
        if not models:
            return {'error': 'No models provided'}

        all_performances = []
        for function_name, model_info in models.items():
            if 'performance' in model_info and 'error' not in model_info:
                perf = model_info['performance']
                all_performances.append({
                    'function_name': function_name,
                    'mse': perf['mse'],
                    'mae': perf['mae'],
                    'r2': perf['r2']
                })

        if not all_performances:
            return None

        def combined_score(perf):
            mse_norm = (perf['mse'] - min(p['mse'] for p in all_performances)) / \
                      (max(p['mse'] for p in all_performances) - min(p['mse'] for p in all_performances))
            mae_norm = (perf['mae'] - min(p['mae'] for p in all_performances)) / \
                      (max(p['mae'] for p in all_performances) - min(p['mae'] for p in all_performances))
            r2_norm = (max(p['r2'] for p in all_performances) - perf['r2']) / \
                      (max(p['r2'] for p in all_performances) - min(p['r2'] for p in all_performances))
            return (mse_norm + mae_norm - r2_norm) / 3

        best_performance = min(all_performances, key=combined_score)
        best_function_name = best_performance['function_name']
        return models.get(best_function_name)

    def select_best_function_for_type(self, models: dict[str, Any]) -> dict[str, Any]:
        """Select best decay function for specific news type."""
        valid_models = {name: info for name, info in models.items()
                       if 'performance' in info and 'error' not in info}
        if not valid_models:
            return {'error': 'No valid models'}

        best_model = max(valid_models.items(), key=lambda x: x[1]['performance']['r2'])
        return best_model[1]
