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

from src.core.logging.logger import ProjectLogger
from src.features.analysis.decay.decay_function_registry import DecayFunctionRegistry
from src.features.analysis.decay.decay_model_fitter import DecayModelFitter
from src.features.analysis.decay.decay_predictor import DecayPredictor

logger = ProjectLogger.get_logger('NewsDecayModeler')


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

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize NewsDecayModeler.

        Args:
            config: Configuration dictionary for decay modeling
        """
        self.logger = logger
        self.config = config or {}

        # Initialize modular components
        self.decay_registry = DecayFunctionRegistry()
        self.model_fitter = DecayModelFitter(
            optimization_enabled=self.config.get('optimization_enabled', True)
        )
        self.decay_predictor = DecayPredictor(
            self.decay_registry.get_all_decay_functions()
        )

        # Configuration
        self.validation_window_hours = self.config.get('validation_window_hours', 72)
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/news_decay'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # State
        self.trained_models: dict[str, Any] = {}
        self.decay_parameters: dict[str, Any] = {}

        self.logger.info('✅ NewsDecayModeler initialized with modular components')

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
        self.logger.info(f'🧠 Fitting optimal decay model for {len(news_data)} news items')

        results: dict[str, Any] = {
            'fitted_models': {},
            'performance_metrics': {},
            'best_overall_model': None,
            'news_type_models': {},
            'optimization_summary': {}
        }

        try:
            prepared_data = await self._prepare_modeling_data(news_data, market_returns, news_type_column)
            if prepared_data['error']:
                return prepared_data

            news_types = self.decay_registry.get_all_news_types()

            for news_type in news_types.keys():
                if news_type in prepared_data['news_by_type']:
                    type_results = await self._fit_news_type_model(
                        prepared_data['news_by_type'][news_type],
                        news_type
                    )
                    results['news_type_models'][news_type] = type_results
                    if isinstance(results['fitted_models'], dict) and isinstance(type_results.get('models'), dict):
                        results['fitted_models'].update(type_results['models'])

            best_model = self.model_fitter.select_best_overall_model(results['fitted_models'] or {})
            results['best_overall_model'] = best_model
            results['optimization_summary'] = await self._generate_optimization_summary(results)

            if isinstance(results['fitted_models'], dict):
                self.trained_models.update(results['fitted_models'])

            self._store_trained_models(results)

            self.logger.info(f"✅ Optimal decay model fitting complete. Best model: {best_model['model_name']}")
            return results
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error fitting optimal decay model: {e}', exc_info=True)
            return {'status': 'error', 'error': str(e)}

    async def _prepare_modeling_data(self,
                                    news_data: pd.DataFrame,
                                    market_returns: pd.DataFrame,
                                    news_type_column: str) -> dict[str, Any]:
        """Prepare data for decay modeling."""
        try:
            if 'timestamp' not in news_data.columns or 'impact_score' not in news_data.columns:
                return {'error': 'Missing required columns in news_data'}
            if 'timestamp' not in market_returns.columns or 'returns' not in market_returns.columns:
                return {'error': 'Missing required columns in market_returns'}

            news_data['timestamp'] = pd.to_datetime(news_data['timestamp'])
            market_returns['timestamp'] = pd.to_datetime(market_returns['timestamp'])

            news_by_type = {}
            if news_type_column in news_data.columns:
                news_types = self.decay_registry.get_all_news_types()
                for news_type in news_types.keys():
                    type_news = news_data[news_data[news_type_column] == news_type].copy()
                    if not type_news.empty:
                        news_by_type[news_type] = type_news

            aligned_data = []
            for _, news_row in news_data.iterrows():
                news_time = news_row['timestamp']
                news_impact = news_row['impact_score']
                time_window = timedelta(hours=self.validation_window_hours)
                start_time = news_time - time_window
                end_time = news_time + time_window
                relevant_returns = market_returns[
                    (market_returns['timestamp'] >= start_time) &
                    (market_returns['timestamp'] <= end_time)
                ].copy()

                if not relevant_returns.empty:
                    relevant_returns['hours_since_news'] = (relevant_returns['timestamp'] - news_time).dt.total_seconds() / 3600
                    relevant_returns['news_impact'] = news_impact
                    aligned_data.append(relevant_returns)

            if not aligned_data:
                return {'error': 'No aligned data found between news and market returns'}

            combined_data = pd.concat(aligned_data, ignore_index=True)
            return {'combined_data': combined_data, 'news_by_type': news_by_type, 'error': None}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Error preparing modeling data: {e}')
            return {'error': str(e)}

    async def _fit_news_type_model(self, news_data: pd.DataFrame, news_type: str) -> dict[str, Any]:
        """Fit decay model for specific news type."""
        type_results = {'news_type': news_type, 'models': {}, 'best_model': None, 'performance_comparison': {}}

        try:
            news_config = self.decay_registry.get_news_type_config(news_type)
            news_config['typical_function']
            typical_hours = float(news_config['typical_decay_hours'])

            decay_functions = self.decay_registry.get_all_decay_functions()

            for function_name, function_config in decay_functions.items():
                model_result = self.model_fitter.fit_decay_function(
                    news_data, function_name, function_config, typical_hours
                )

                if isinstance(type_results.get('models'), dict):
                    type_results['models'][function_name] = model_result

                if isinstance(type_results.get('performance_comparison'), dict):
                    type_results['performance_comparison'][function_name] = model_result.get('performance', {})

            best_model = self.model_fitter.select_best_function_for_type(type_results.get('models', {}))
            type_results['best_model'] = best_model

            models_count = len(type_results.get('models', {}))
            self.logger.info(f'📊 Fitted {models_count} decay models for {news_type}')

            return type_results
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Error fitting news type model for {news_type}: {e}')
            return {'news_type': news_type, 'models': {}, 'best_model': None, 'performance_comparison': {}, 'error': str(e)}

    async def _generate_optimization_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive optimization summary."""
        summary: dict[str, Any] = {
            'total_models_fitted': len(results.get('fitted_models', {})),
            'news_types_analyzed': len(results.get('news_type_models', {})),
            'best_overall_function': results.get('best_overall_model', {}).get('function_name'),
            'performance_ranking': {},
            'recommendations': []
        }

        try:
            all_models = []
            for function_name, model_info in results['fitted_models'].items():
                if 'performance' in model_info:
                    perf = model_info['performance']
                    all_models.append({
                        'function_name': function_name,
                        'mse': perf['mse'],
                        'mae': perf['mae'],
                        'r2': perf['r2']
                    })

            all_models.sort(key=lambda x: x['r2'], reverse=True)

            summary['performance_ranking'] = {
                'top_models': all_models[:5],
                'worst_models': all_models[-3:] if len(all_models) > 3 else []
            }

            if all_models:
                best_r2 = all_models[0]['r2']
                if best_r2 > 0.7:
                    summary['recommendations'].append(
                        '✅ Excellent model fit achieved (R2 > 0.7). Current decay modeling is highly effective.'
                    )
                elif best_r2 > 0.5:
                    summary['recommendations'].append(
                        '⚠️ Good model fit (R2 > 0.5). Consider fine-tuning parameters.'
                    )
                else:
                    summary['recommendations'].append(
                        '❌ Poor model fit (R2 < 0.5). Consider alternative decay functions.'
                    )

                best_function = all_models[0]['function_name']
                if best_function == 'exponential':
                    summary['recommendations'].append(
                        '📈 Exponential decay performs best. Consider using for most news types.'
                    )
                elif best_function == 'step_function':
                    summary['recommendations'].append(
                        '📊 Step function performs best. Suitable for macro news with immediate impact.'
                    )
                else:
                    summary['recommendations'].append(
                        '❌ Poor model fit. Consider alternative decay functions.'
                    )

            return summary
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Error generating optimization summary: {e}')
            return summary

    def _store_trained_models(self, results: dict[str, Any]) ->None:
        """Store trained decay models for future use."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'decay_models_{timestamp}.json'
            filepath = self.storage_path / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            files = list(self.storage_path.glob('decay_models_*.json'))
            for file_to_delete in files[50:]:
                file_to_delete.unlink()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Failed to store trained models: {e}')

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
        return self.decay_predictor.predict_news_impact(
            news_data,
            self.trained_models,
            model_name,
            news_type,
            self.decay_parameters
        )

    def get_model_performance_summary(self, days: int=30) ->dict[str, Any]:
        """Get summary of model performance over time."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            model_files = list(self.storage_path.glob('decay_models_*.json'))
            recent_files = [f for f in model_files if datetime.
                fromtimestamp(f.stat().st_mtime) >= cutoff_time]
            if not recent_files:
                return {'error': 'No recent model performance data available'}
            performance_history = []
            for file_path in recent_files:
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    if 'best_overall_model' in data and data[
                        'best_overall_model']:
                        perf = data['best_overall_model'].get('performance', {}
                            )
                        file_time = datetime.fromtimestamp(file_path.stat()
                            .st_mtime)
                        performance_history.append({'timestamp': file_time,
                            'model_name': data['best_overall_model'].get(
                            'function_name'), 'mse': perf.get('mse'), 'mae':
                            perf.get('mae'), 'r2': perf.get('r2')})
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.exception(
                        f'Error loading model file {file_path}: {e}')
            if not performance_history:
                return {'error': 'No valid performance data found'}
            performance_trends = {}
            for metric in ['mse', 'mae', 'r2']:
                values = [p[metric] for p in performance_history if metric in
                    p and p[metric] is not None]
                if len(values) >= 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    performance_trends[metric] = {'trend': 'improving' if
                        trend < 0 else 'degrading', 'trend_slope': trend,
                        'latest_value': values[-1], 'average_value': np.
                        mean(values)}
            return {'period_days': days, 'performance_history':
                performance_history, 'performance_trends':
                performance_trends, 'model_stability': self.
                _calculate_model_stability(performance_history)}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Error getting model performance summary: {e}')
            return {'error': str(e)}

    def _calculate_model_stability(self, performance_history: list[dict[str,
        Any]]) ->dict[str, float]:
        """Calculate model stability metrics."""
        if len(performance_history) < 2:
            return {}
        stability_metrics = {}
        for metric in ['mse', 'mae', 'r2']:
            values = [p[metric] for p in performance_history if metric in p and
                p[metric] is not None]
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                stability_metrics[f'{metric}_stability'] = 1.0 / (1 + cv)
        return stability_metrics


def get_news_decay_modeler(config: dict[str, Any] | None=None
    ) ->NewsDecayModeler:
    """Factory function to get NewsDecayModeler instance."""
    return NewsDecayModeler(config)


async def fit_news_decay_model_quick(news_data: pd.DataFrame,
    market_returns: pd.DataFrame, config: dict[str, Any] | None=None
    ) ->dict[str, Any]:
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
