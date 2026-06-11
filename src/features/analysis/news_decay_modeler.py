#!/usr/bin/env python3
"""
News Decay Modeler - ML-Optimized News Impact Decay Modeling
Optimizes news impact decay functions using machine learning techniques.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("NewsDecayModeler")

class NewsDecayModeler:
    """
    ML-based news impact decay optimization and modeling.

    This modeler optimizes news impact decay through:
    - Multiple decay function types (exponential, linear, logarithmic, step)
    - Hyperparameter optimization for decay rates
    - News-type specific decay modeling
    - Time-decay validation against actual market impact
    - Automatic model selection based on performance

    Critical for accurate news impact assessment in trading systems.
    """

    # Decay function types
    DECAY_FUNCTIONS = {
        'exponential': {
            'description': 'Exponential decay with half-life parameter',
            'params': ['half_life_hours', 'initial_impact'],
            'function': lambda t, hl, init: init * np.exp(-np.log(2) * t / hl) if hl > 0 else init
        },
        'linear': {
            'description': 'Linear decay over time',
            'params': ['max_impact_hours', 'initial_impact'],
            'function': lambda t, max_h, init: max(0, init * (1 - t / max_h) if max_h > 0 else init)
        },
        'logarithmic': {
            'description': 'Logarithmic decay with scale factor',
            'params': ['scale_factor', 'initial_impact'],
            'function': lambda t, scale, init: init / (1 + t / scale) if scale > 0 else init
        },
        'step_function': {
            'description': 'Step function with immediate impact and gradual decay',
            'params': ['immediate_hours', 'decay_rate', 'initial_impact'],
            'function': lambda t, imm_h, rate, init: init if t < imm_h else init * np.exp(-rate * (t - imm_h))
        },
        'power_law': {
            'description': 'Power law decay with exponent parameter',
            'params': ['exponent', 'scale_hours', 'initial_impact'],
            'function': lambda t, exp, scale, init: init / (1 + (t / scale) ** exp) if scale > 0 else init
        }
    }

    # News type classifications
    NEWS_TYPES = {
        'earnings': {
            'description': 'Earnings announcements',
            'typical_decay_hours': 48,
            'typical_function': 'exponential',
            'impact_duration': 'long'
        },
        'macro': {
            'description': 'Macroeconomic data releases',
            'typical_decay_hours': 24,
            'typical_function': 'step_function',
            'impact_duration': 'medium'
        },
        'sector': {
            'description': 'Sector-specific news',
            'typical_decay_hours': 12,
            'typical_function': 'exponential',
            'impact_duration': 'medium'
        },
        'company_specific': {
            'description': 'Company-specific news',
            'typical_decay_hours': 6,
            'typical_function': 'linear',
            'impact_duration': 'short'
        },
        'market_sentiment': {
            'description': 'General market sentiment',
            'typical_decay_hours': 8,
            'typical_function': 'logarithmic',
            'impact_duration': 'short'
        }
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize NewsDecayModeler.

        Args:
            config: Configuration dictionary for decay modeling
        """
        self.logger = logger
        self.config = config or {}

        # Model settings
        self.optimization_enabled = self.config.get('optimization_enabled', True)
        self.auto_function_selection = self.config.get('auto_function_selection', True)
        self.validation_window_hours = self.config.get('validation_window_hours', 72)

        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/news_decay'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Trained models storage
        self.trained_models: dict[str, Any] = {}
        self.decay_parameters: dict[str, Any] = {}

        self.logger.info("✅ NewsDecayModeler initialized")

    async def fit_optimal_decay_model(self,
                                   news_data: pd.DataFrame,
                                   market_returns: pd.DataFrame,
                                   news_type_column: str = 'news_type') -> dict[str, Any]:
        """
        Fit optimal decay model using machine learning optimization.

        Args:
            news_data: DataFrame with news timestamps and impact scores
            market_returns: DataFrame with actual market returns
            news_type_column: Column name for news type classification

        Returns:
            Dict with fitted decay models and performance metrics
        """
        self.logger.info(f"🧠 Fitting optimal decay model for {len(news_data)} news items")

        results: dict[str, Any] = {
            'fitted_models': {},
            'performance_metrics': {},
            'best_overall_model': None,
            'news_type_models': {},
            'optimization_summary': {}
        }

        try:
            # 1. Prepare data for modeling
            prepared_data = await self._prepare_modeling_data(news_data, market_returns, news_type_column)

            if prepared_data['error']:
                return prepared_data

            # 2. Fit models for each news type
            for news_type in self.NEWS_TYPES.keys():
                if news_type in prepared_data['news_by_type']:
                    type_results = await self._fit_news_type_model(
                        prepared_data['news_by_type'][news_type],
                        news_type
                    )
                    results['news_type_models'][news_type] = type_results
                    if isinstance(results['fitted_models'], dict) and isinstance(type_results.get('models'), dict):
                        results['fitted_models'].update(type_results['models'])

            # 3. Find best overall model
            best_model = await self._select_best_overall_model(results['fitted_models'] or {})
            results['best_overall_model'] = best_model

            # 4. Generate optimization summary
            results['optimization_summary'] = await self._generate_optimization_summary(results)

            # 5. Store trained models
            if isinstance(results['fitted_models'], dict):
                self.trained_models.update(results['fitted_models'])
            self._store_trained_models(results)

            self.logger.info(f"✅ Optimal decay model fitting complete. Best model: {best_model['model_name']}")

            return results

        except Exception as e:
            self.logger.error(f"Error fitting optimal decay model: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _prepare_modeling_data(self,
                                  news_data: pd.DataFrame,
                                  market_returns: pd.DataFrame,
                                  news_type_column: str) -> dict[str, Any]:
        """Prepare data for decay modeling."""

        try:
            # Validate inputs
            if 'timestamp' not in news_data.columns or 'impact_score' not in news_data.columns:
                return {'error': 'Missing required columns in news_data'}

            if 'timestamp' not in market_returns.columns or 'returns' not in market_returns.columns:
                return {'error': 'Missing required columns in market_returns'}

            # Convert timestamps
            news_data['timestamp'] = pd.to_datetime(news_data['timestamp'])
            market_returns['timestamp'] = pd.to_datetime(market_returns['timestamp'])

            # Group news by type
            news_by_type = {}
            if news_type_column in news_data.columns:
                for news_type in self.NEWS_TYPES.keys():
                    type_news = news_data[news_data[news_type_column] == news_type].copy()
                    if not type_news.empty:
                        news_by_type[news_type] = type_news

            # Align news data with market returns
            aligned_data = []

            for _, news_row in news_data.iterrows():
                news_time = news_row['timestamp']
                news_impact = news_row['impact_score']

                # Find market returns around news time
                time_window = timedelta(hours=self.validation_window_hours)
                start_time = news_time - time_window
                end_time = news_time + time_window

                relevant_returns = market_returns[
                    (market_returns['timestamp'] >= start_time) &
                    (market_returns['timestamp'] <= end_time)
                ].copy()

                if not relevant_returns.empty:
                    # Calculate time differences
                    relevant_returns['hours_since_news'] = (
                        relevant_returns['timestamp'] - news_time
                    ).dt.total_seconds() / 3600

                    # Add news impact to returns data
                    relevant_returns['news_impact'] = news_impact

                    aligned_data.append(relevant_returns)

            if not aligned_data:
                return {'error': 'No aligned data found between news and market returns'}

            combined_data = pd.concat(aligned_data, ignore_index=True)

            return {
                'combined_data': combined_data,
                'news_by_type': news_by_type,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Error preparing modeling data: {e}")
            return {'error': str(e)}

    async def _fit_news_type_model(self,
                                 news_data: pd.DataFrame,
                                 news_type: str) -> dict[str, Any]:
        """Fit decay model for specific news type."""

        type_results = {
            'news_type': news_type,
            'models': {},
            'best_model': None,
            'performance_comparison': {}
        }

        try:
            # Get typical parameters for this news type
            news_config = self.NEWS_TYPES[news_type]
            news_config['typical_function']
            typical_hours = float(news_config['typical_decay_hours'])  # type: ignore

            # Fit each decay function
            for function_name, function_config in self.DECAY_FUNCTIONS.items():
                model_result = await self._fit_decay_function(
                    news_data, function_name, function_config, typical_hours
                )

                if isinstance(type_results.get('models'), dict):
                    type_results['models'][function_name] = model_result  # type: ignore
                if isinstance(type_results.get('performance_comparison'), dict):
                    type_results['performance_comparison'][function_name] = model_result.get('performance', {})  # type: ignore

            # Select best model for this news type
            best_model = await self._select_best_function_for_type(type_results.get('models', {}))  # type: ignore
            type_results['best_model'] = best_model

            models_count = len(type_results.get('models', {}))  # type: ignore
            self.logger.info(f"📊 Fitted {models_count} decay models for {news_type}")

            return type_results

        except Exception as e:
            self.logger.error(f"Error fitting news type model for {news_type}: {e}")
            return type_results  # type: ignore

    async def _fit_decay_function(self,
                              news_data: pd.DataFrame,
                              function_name: str,
                              function_config: dict[str, Any],
                              typical_hours: float) -> dict[str, Any]:
        """Fit a specific decay function to news data."""

        try:
            # Prepare training data
            X = news_data[['hours_since_news']].values
            y = news_data['returns'].values

            # Define decay function with parameters
            def decay_function_with_params(params, hours):
                return function_config['function'](hours, *params)

            # Initial parameter guess
            initial_params = self._get_initial_parameters(function_name, typical_hours)

            # Optimize parameters using scipy
            def objective(params):
                try:
                    predicted_impact = decay_function_with_params(params, X.flatten())
                    mse = mean_squared_error(y, predicted_impact)
                    return mse
                except:
                    return float('inf')

            # Run optimization
            if self.optimization_enabled:
                result = minimize(
                    objective,
                    initial_params,
                    method='L-BFGS',
                    options={'maxiter': 100}
                )

                if result.success:
                    optimal_params = result.x
                    optimization_success = True
                else:
                    optimal_params = initial_params
                    optimization_success = False
            else:
                optimal_params = initial_params
                optimization_success = True

            # Calculate performance metrics
            predicted_impact = decay_function_with_params(optimal_params, X.flatten())

            performance = {
                'mse': mean_squared_error(y, predicted_impact),
                'mae': mean_absolute_error(y, predicted_impact),
                'r2': self._calculate_r2(y, predicted_impact),
                'optimization_success': optimization_success
            }

            # Create model info
            model_info = {
                'function_name': function_name,
                'parameters': dict(zip(function_config['params'], optimal_params, strict=False)),
                'performance': performance,
                'description': function_config['description']
            }

            return model_info

        except Exception as e:
            self.logger.error(f"Error fitting decay function {function_name}: {e}")
            return {
                'function_name': function_name,
                'error': str(e)
            }

    def _get_initial_parameters(self, function_name: str, typical_hours: float) -> list[float]:
        """Get initial parameter guess for optimization."""

        if function_name == 'exponential':
            return [typical_hours, 1.0]  # half_life, initial_impact
        elif function_name == 'linear':
            return [typical_hours * 2, 1.0]  # max_hours, initial_impact
        elif function_name == 'logarithmic':
            return [typical_hours, 1.0]  # scale, initial_impact
        elif function_name == 'step_function':
            return [1.0, 0.1, 1.0]  # immediate_hours, decay_rate, initial_impact
        elif function_name == 'power_law':
            return [1.0, typical_hours, 1.0]  # exponent, scale, initial_impact
        else:
            return [1.0, 1.0]  # Default

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 1.0

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)

    async def _select_best_overall_model(self, models: dict[str, Any]) -> dict[str, Any]:
        """Select best overall model across all news types."""

        if not models:
            return {'error': 'No models provided'}

        # Collect all model performances
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
            return None  # type: ignore

        # Select best model based on combined score
        def combined_score(perf):
            # Normalize metrics (lower is better for MSE/MAE, higher for R2)
            mse_norm = (perf['mse'] - min(p['mse'] for p in all_performances)) / (
                max(p['mse'] for p in all_performances) - min(p['mse'] for p in all_performances)
            )
            mae_norm = (perf['mae'] - min(p['mae'] for p in all_performances)) / (
                max(p['mae'] for p in all_performances) - min(p['mae'] for p in all_performances)
            )
            r2_norm = (max(p['r2'] for p in all_performances) - perf['r2']) / (
                max(p['r2'] for p in all_performances) - min(p['r2'] for p in all_performances)
            )

            # Combined score (lower is better)
            return (mse_norm + mae_norm - r2_norm) / 3

        best_performance = min(all_performances, key=combined_score)
        best_function_name = best_performance['function_name']

        return models.get(best_function_name)  # type: ignore

    async def _select_best_function_for_type(self, models: dict[str, Any]) -> dict[str, Any]:
        """Select best decay function for specific news type."""

        valid_models = {
            name: info for name, info in models.items()
            if 'performance' in info and 'error' not in info
        }

        if not valid_models:
            return {'error': 'No valid models'}

        # Select based on R2 score (higher is better)
        best_model = max(valid_models.items(), key=lambda x: x[1]['performance']['r2'])

        return best_model[1]  # type: ignore

    async def _generate_optimization_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive optimization summary."""

        summary: dict[str, Any] = {
            'total_models_fitted': len(results.get('fitted_models', {})),
            'news_types_analyzed': len(results.get('news_type_models', {})),
            'best_overall_function': results.get('best_overall_model', {}).get('function_name'),
            'performance_ranking': {},
            'recommendations': []
        }  # type: ignore

        try:
            # Rank all models by performance
            all_models = []

            for function_name, model_info in results['fitted_models'].items():
                if 'performance' in model_info:
                    perf = model_info['performance']
                    all_models.append({
                        'function_name': function_name,
                        'mse': perf['mse'],
                        'mae': perf['mae'],
                        'r2': perf['r2']
                    })  # type: ignore

            # Sort by R2 score
            all_models.sort(key=lambda x: x['r2'], reverse=True)  # type: ignore

            summary['performance_ranking'] = {
                'top_models': all_models[:5],
                'worst_models': all_models[-3:] if len(all_models) > 3 else []
            }

            # Generate recommendations
            if all_models:
                best_r2 = all_models[0]['r2']

                if best_r2 > 0.7:
                    summary['recommendations'].append(
                        "✅ Excellent model fit achieved (R2 > 0.7). Current decay modeling is highly effective."
                    )
                elif best_r2 > 0.5:
                    summary['recommendations'].append(
                        "⚠️ Good model fit (R2 > 0.5). Consider fine-tuning parameters."
                    )
                else:
                    summary['recommendations'].append(
                        "❌ Poor model fit (R2 < 0.5). Consider alternative decay functions."
                    )

                # Function-specific recommendations
                best_function = all_models[0]['function_name']
                if best_function == 'exponential':
                    summary['recommendations'].append(
                        "📈 Exponential decay performs best. Consider using for most news types."
                    )
                elif best_function == 'step_function':
                    summary['recommendations'].append(
                        "📊 Step function performs best. Suitable for macro news with immediate impact."
                    )
                else:
                    summary['recommendations'].append(
                        "❌ Poor model fit. Consider alternative decay functions."
                    )

            return summary

        except Exception as e:
            self.logger.error(f"Error generating optimization summary: {e}")
            return summary  # type: ignore

    def _store_trained_models(self, results: dict[str, Any]) -> None:
        """Store trained decay models for future use."""

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"decay_models_{timestamp}.json"
            filepath = self.storage_path / filename

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Keep only last 50 model files
            files = list(self.storage_path.glob("decay_models_*.json"))

            for file_to_delete in files[50:]:
                file_to_delete.unlink()

        except Exception as e:
            self.logger.error(f"Failed to store trained models: {e}")

    async def predict_news_impact(self,
                               news_data: pd.DataFrame,
                               model_name: str | None = None,
                               news_type: str | None = None) -> pd.DataFrame:
        """
        Predict news impact using trained decay models.

        Args:
            news_data: DataFrame with news timestamps
            model_name: Specific decay model to use (uses best if None)
            news_type: News type for model selection

        Returns:
            DataFrame with predicted impact over time
        """

        try:
            if not self.trained_models:
                self.logger.warning("No trained models available. Please fit models first.")
                return pd.DataFrame()

            # Select model to use
            if model_name and model_name in self.trained_models:
                model_info = self.trained_models[model_name]
            elif news_type and news_type in self.decay_parameters:
                model_info = self.decay_parameters[news_type]
            else:
                # Use best overall model
                model_info = self.trained_models.get('best_overall_model', {})

            if not model_info or 'parameters' not in model_info:
                self.logger.error("No valid model found for prediction")
                return pd.DataFrame()

            # Generate impact predictions
            predictions = []

            function_config = self.DECAY_FUNCTIONS.get(model_info.get('function_name'))
            if not function_config:
                return pd.DataFrame()

            for _, news_row in news_data.iterrows():
                news_time = pd.to_datetime(news_row['timestamp'])  # type: ignore
                hours_range = np.arange(0, 73, 1)  # 0 to 72 hours

                # Apply decay function to get impact for each hour
                impact_predictions = function_config['function'](
                    hours_range,
                    *list(model_info['parameters'].values())
                )  # type: ignore

                # Create prediction records
                for hour, impact in zip(hours_range, impact_predictions, strict=False):
                    prediction_time = news_time + timedelta(hours=hour)

                    predictions.append({
                        'news_timestamp': news_time,
                        'prediction_timestamp': prediction_time,
                        'hours_since_news': hour,
                        'predicted_impact': impact,
                        'model_name': model_info.get('function_name'),
                        'news_type': news_type
                    })

            return pd.DataFrame(predictions)

        except Exception as e:
            self.logger.error(f"Error predicting news impact: {e}")
            return pd.DataFrame()

    def get_model_performance_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of model performance over time."""

        try:
            # Load recent model files
            cutoff_time = datetime.now() - timedelta(days=days)

            model_files = list(self.storage_path.glob("decay_models_*.json"))
            recent_files = [
                f for f in model_files
                if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff_time
            ]

            if not recent_files:
                return {'error': 'No recent model performance data available'}

            # Analyze performance trends
            performance_history = []

            for file_path in recent_files:
                try:
                    with open(file_path) as f:
                        data = json.load(f)

                    if 'best_overall_model' in data and data['best_overall_model']:
                        perf = data['best_overall_model'].get('performance', {})
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)

                        performance_history.append({
                            'timestamp': file_time,
                            'model_name': data['best_overall_model'].get('function_name'),
                            'mse': perf.get('mse'),
                            'mae': perf.get('mae'),
                            'r2': perf.get('r2')
                        })
                except Exception as e:
                    self.logger.error(f"Error loading model file {file_path}: {e}")

            if not performance_history:
                return {'error': 'No valid performance data found'}

            # Calculate trends
            performance_trends = {}
            for metric in ['mse', 'mae', 'r2']:
                values = [p[metric] for p in performance_history if metric in p and p[metric] is not None]

                if len(values) >= 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    performance_trends[metric] = {
                        'trend': 'improving' if trend < 0 else 'degrading',
                        'trend_slope': trend,
                        'latest_value': values[-1],
                        'average_value': np.mean(values)
                    }

            return {
                'period_days': days,
                'performance_history': performance_history,
                'performance_trends': performance_trends,
                'model_stability': self._calculate_model_stability(performance_history)
            }

        except Exception as e:
            self.logger.error(f"Error getting model performance summary: {e}")
            return {'error': str(e)}

    def _calculate_model_stability(self, performance_history: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate model stability metrics."""

        if len(performance_history) < 2:
            return {}

        # Calculate coefficient of variation for each metric
        stability_metrics = {}

        for metric in ['mse', 'mae', 'r2']:
            values = [p[metric] for p in performance_history if metric in p and p[metric] is not None]

            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float('inf')

                stability_metrics[f'{metric}_stability'] = 1.0 / (1 + cv)  # Higher is more stable

        return stability_metrics


# Factory function for easy instantiation
def get_news_decay_modeler(config: dict[str, Any] | None = None) -> NewsDecayModeler:
    """Factory function to get NewsDecayModeler instance."""
    return NewsDecayModeler(config)


# Convenience function for quick fitting
async def fit_news_decay_model_quick(news_data: pd.DataFrame,
                                  market_returns: pd.DataFrame,
                                  config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Quick news decay model fitting.

    Args:
        news_data: News data with timestamps and impact scores
        market_returns: Market returns data
        config: Configuration dictionary

    Returns:
        Fitted decay model results
    """
    modeler = get_news_decay_modeler(config)
    return await modeler.fit_optimal_decay_model(news_data, market_returns)
