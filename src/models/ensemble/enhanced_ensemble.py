"""
Enhanced Ensemble Model - Refactored Version
Reduced cognitive complexity by extracting helper methods
"""
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
logger = logging.getLogger(__name__)


class EnhancedEnsembleModel:
    """Enhanced ensemble model with improved architecture"""

    def __init__(self, models_path: str='data/models'):
        self.models_path = Path(models_path)
        self.heavy_models = {}
        self.light_models = {}
        self.diary = None
        self.logger = logger

    def load_colab_results(self, colab_results: Dict[str, Any]) ->Dict[str, Any
        ]:
        """Load results of heavy models from Colab"""
        logger.info('📥 Loading heavy models from Colab...')
        heavy_models = {}
        models_metadata = colab_results.get('models_metadata', {})
        batch_name = colab_results.get('batch_name', 'main_database')
        if models_metadata:
            heavy_models = self._process_models_metadata(models_metadata,
                batch_name)
        else:
            heavy_models = self._process_timeframe_format(colab_results)
        self.heavy_models = heavy_models
        logger.info(f'✅ Loaded {len(heavy_models)} heavy models')
        return heavy_models

    def _process_models_metadata(self, models_metadata: Dict, batch_name: str
        ) ->Dict[str, Any]:
        """Process models metadata format"""
        heavy_models = {}
        for model_key, model_data in models_metadata.items():
            if not isinstance(model_data, dict) or model_data.get('type'
                ) != 'heavy':
                continue
            model_path = model_data.get('model_path', '')
            model_obj = self._load_model_from_path(model_path, batch_name)
            heavy_models[model_key] = {'ticker': model_data.get('ticker'),
                'target': model_data.get('target'), 'timeframe': model_data
                .get('timeframe'), 'model_type': model_data.get(
                'model_type'), 'mse': model_data.get('mse', 0), 'type':
                'heavy', 'source': 'colab', 'model_path': model_path,
                'status': model_data.get('status', 'unknown'), 'model_obj':
                model_obj}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"✅ Added heavy model: {model_key} (MSE: {model_data.get('mse', 0):.6f})"
                    )
        return heavy_models

    def _process_timeframe_format(self, colab_results: Dict) ->Dict[str, Any]:
        """Process timeframe format results"""
        heavy_models = {}
        ticker = colab_results.get('ticker')
        timeframes = colab_results.get('timeframes', {})
        if not ticker or not timeframes:
            return heavy_models
        return self._process_timeframes_data(timeframes, ticker)

    def _process_timeframes_data(self, timeframes: Dict, ticker: str) ->Dict[
        str, Any]:
        """Process timeframes data"""
        heavy_models = {}
        for timeframe, tf_data in timeframes.items():
            results = tf_data.get('results', {})
            for target_name, target_result in results.items():
                if not isinstance(target_result, dict
                    ) or 'models' not in target_result:
                    continue
                models = target_result.get('models', {})
                for model_type, model_data in models.items():
                    if not isinstance(model_data, dict
                        ) or 'mse' not in model_data:
                        continue
                    model_key = f'{ticker}_{target_name}_{model_type}'
                    model_path = model_data.get('model_path', '')
                    model_obj = self._load_model_from_path(model_path,
                        'main_database')
                    heavy_models[model_key] = {'ticker': ticker, 'target':
                        target_name, 'timeframe': timeframe, 'model_type':
                        model_type, 'mse': model_data.get('mse', 0), 'type':
                        'heavy', 'source': 'colab', 'model_path':
                        model_path, 'model_obj': model_obj}
        return heavy_models

    def _load_model_from_path(self, model_path: str, batch_name: str
        ) ->Optional[Any]:
        """Load model from various possible paths with security validation"""
        if not model_path:
            return None
        
        # Security validation: Ensure path is within expected data directories
        def is_safe_path(p: str) -> bool:
            abs_p = Path(p).resolve()
            allowed_bases = [
                Path('data').resolve(),
                Path('models').resolve()
            ]
            return any(abs_p.is_relative_to(base) for base in allowed_bases)

        possible_paths = [model_path]
        model_filename = Path(model_path).name
        possible_paths.extend([
            f'data/colab/accumulated/{batch_name}/models/{model_filename}',
            f'data/colab/accumulated/test_ticker_amd_target_1d_ep5_iter5/models/{model_filename}'
            , f'data/colab/accumulated/main_database/models/{model_filename}'])
        
        for path_candidate in possible_paths:
            if Path(path_candidate).exists():
                if not is_safe_path(path_candidate):
                    logger.warning(f"🚫 Blocking unsafe model load attempt from: {path_candidate}")
                    continue
                try:
                    import torch

                    # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
                    model_obj = torch.load(  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
                        path_candidate, map_location='cpu', weights_only=True)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'✅ Loaded model from {path_candidate}')
                    return model_obj
                except Exception as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f'⚠️ Failed to load model {path_candidate}: {e}')
                    raise
        return None

    def _should_skip_model(self, model_info: Dict, target_cols: List[str],
        tickers: List[str], timeframes: List[str]) ->bool:
        """Check if model should be skipped based on filters"""
        if target_cols and model_info['target'] not in target_cols:
            return True
        if tickers and model_info['ticker'] not in tickers:
            return True
        if timeframes and model_info['timeframe'] not in timeframes:
            return True
        return False

    def _load_single_model(self, model_file: Path, light_model_types: List[str]
        ) ->Optional[Dict[str, Any]]:
        """Load a single model file with security validation"""
        try:
            # Security validation: Ensure path is within expected models directory
            abs_p = model_file.resolve()
            allowed_base = self.models_path.resolve()
            if not abs_p.is_relative_to(allowed_base):
                logger.warning(f"🚫 Blocking unsafe model load attempt from outside models directory: {model_file}")
                return None

            # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            model = joblib.load(model_file)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            stem = model_file.stem
            model_info = self._parse_model_filename(stem)
            if not model_info or model_info['model_type'
                ] not in light_model_types:
                return None
            return {'ticker': model_info['ticker'], 'target': model_info[
                'target'], 'timeframe': model_info['timeframe'],
                'model_type': model_info['model_type'], 'model_path': str(
                model_file), 'model_obj': model, 'type': 'light'}
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'⚠️ Failed to load {model_file}: {e}')
            raise RuntimeError(f"Failed to load model file {model_file}") from e

    def load_local_light_models(self, target_cols: List[str]=None, tickers:
        List[str]=None, timeframes: List[str]=None) ->Dict[str, Any]:
        """Load ALL 8 types of light models locally considering timeframe"""
        logger.info(
            '📥 Loading ALL 8 types of light models locally (with TF support)...'
            )
        light_models = {}
        light_model_types = ['catboost', 'lightgbm', 'xgboost',
            'random_forest', 'linear', 'svm', 'knn', 'ensemble']
        model_files = list(self.models_path.glob('*.joblib'))
        logger.info(f'🔍 Found {len(model_files)} model files. Filtering...')
        for model_file in model_files:
            model_data = self._load_single_model(model_file, light_model_types)
            if not model_data:
                continue
            model_info = self._parse_model_filename(model_file.stem)
            if self._should_skip_model(model_info, target_cols, tickers,
                timeframes):
                continue
            model_key = (
                f"{model_data['ticker']}_{model_data['target']}_{model_data['model_type']}"
                )
            light_models[model_key] = model_data
        self.light_models = light_models
        self._log_model_statistics(light_models)
        return light_models

    def _parse_model_filename(self, stem: str) ->Optional[Dict[str, str]]:
        """Parse model filename to extract components"""
        ticker = None
        timeframe = None
        target = None
        model_type = None
        if stem.startswith('CHAMP_'):
            parts = stem[6:].split('_')
            if len(parts) >= 3:
                ticker = parts[0]
                timeframe = parts[1]
                target = '_'.join(parts[2:])
                model_type = 'champion'
        else:
            parts = stem.split('_')
            if len(parts) >= 4:
                ticker = parts[0]
                timeframe = parts[1]
                target = parts[2]
                model_type = '_'.join(parts[3:])
        if all([ticker, timeframe, target, model_type]):
            return {'ticker': ticker, 'timeframe': timeframe, 'target':
                target, 'model_type': model_type}
        return None

    def _log_model_statistics(self, light_models: Dict[str, Any]):
        """Log model statistics"""
        timeframes = set()
        for key, info in light_models.items():
            timeframes.add(info.get('timeframe'))
        if timeframes:
            logger.info(f'   Timeframes: {sorted(timeframes)}')
        model_types = {}
        for key, info in light_models.items():
            m_type = info.get('model_type')
            model_types[m_type] = model_types.get(m_type, 0) + 1
        if model_types:
            logger.info('📊 Models by type:')
            for m_type, count in sorted(model_types.items()):
                logger.info(f'   - {m_type}: {count}')

    def get_model_statistics(self) ->Dict[str, Any]:
        """Get model statistics"""
        return {'heavy_models_count': len(self.heavy_models),
            'light_models_count': len(self.light_models), 'total_models': 
            len(self.heavy_models) + len(self.light_models), 'heavy_models':
            list(self.heavy_models.keys())[:10], 'light_models': list(self.
            light_models.keys())[:10]}

    def vectorized_comparison(self, features_df: pd.DataFrame, label_names:
        List[str]) ->Dict[str, Any]:
        """Vectorized comparison of models"""
        logger.info('🔍 Vectorized comparison of models...')
        comparison_results = {}
        for ticker in self._get_unique_tickers(features_df):
            ticker_features = self._get_ticker_features(features_df, ticker)
            for label_name in label_names:
                comparison_key = f'{ticker}_{label_name}'
                comparison_results[comparison_key
                    ] = self._create_comparison_result(ticker, label_name,
                    ticker_features)
        logger.info(f'✅ Compared {len(comparison_results)} model combinations')
        return comparison_results

    def _get_unique_tickers(self, features_df: pd.DataFrame) ->List[str]:
        """Get unique tickers from features dataframe"""
        return features_df['ticker'].unique().tolist()

    def _get_ticker_features(self, features_df: pd.DataFrame, ticker: str
        ) ->pd.DataFrame:
        """Get features for specific ticker"""
        return features_df[features_df['ticker'] == ticker]

    def _create_comparison_result(self, ticker: str, label_name: str,
        ticker_features: pd.DataFrame) ->Dict[str, Any]:
        """Create comparison result for ticker-label combination"""
        if label_name not in ticker_features.columns:
            return None
        label_series = ticker_features[label_name]
        analysis_features = ticker_features.drop(columns=[label_name],
            errors='ignore')
        label_data = label_series.dropna()
        if len(label_data) < 10:
            return None
        return {'ticker': ticker, 'target': label_name, 'heavy_quality': 
            0.5, 'heavy_predictions': [], 'light_predictions': [],
            'light_qualities': []}

    def create_ensemble(self, comparison_results: Dict[str, Any]) ->Dict[
        str, Any]:
        """Create ensemble of heavy + light models with dynamic confidence and KNN correction"""
        logger.info('🎯 Creating heavy + light ensemble with KNN correction...')
        ensemble_results = {}
        for key, comp_result in comparison_results.items():
            ensemble_result = self._create_single_ensemble(comp_result)
            if ensemble_result:
                ensemble_results[key] = ensemble_result
        logger.info(f'✅ Ensemble ready: {len(ensemble_results)} predictions')
        return ensemble_results

    def _create_single_ensemble(self, comp_result: Dict[str, Any]) ->Dict[
        str, Any]:
        """Create ensemble for a single comparison result"""
        ticker = comp_result['ticker']
        target = comp_result['target']
        heavy_quality = comp_result.get('heavy_quality', 0)
        heavy_predictions = comp_result.get('heavy_predictions', [])
        light_predictions = comp_result.get('light_predictions', [])
        light_qualities = comp_result.get('light_qualities', [])
        if (not light_predictions and not heavy_predictions and 
            heavy_quality == 0):
            return None
        if heavy_predictions and light_predictions:
            ensemble_pred = np.mean(heavy_predictions + light_predictions,
                axis=0)
        elif heavy_predictions:
            ensemble_pred = np.mean(heavy_predictions, axis=0)
        elif light_predictions:
            ensemble_pred = np.mean(light_predictions, axis=0)
        else:
            return None
        return {'ticker': ticker, 'target': target, 'prediction':
            ensemble_pred, 'heavy_quality': heavy_quality, 'light_quality':
            np.mean(light_qualities) if light_qualities else 0,
            'heavy_predictions': heavy_predictions, 'light_predictions':
            light_predictions, 'n_heavy': len(heavy_predictions), 'n_light':
            len(light_predictions), 'knn_applied': False, 'n_models': len(
            heavy_predictions) + len(light_predictions)}
