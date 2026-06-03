"""
Model Selection Service for Stage 5 Prediction.

Handles model selection logic including:
- Available model type detection
- Model candidate filtering
- Contextual model selection using selectors
- Model alias resolution
Extracted from stage_5_prediction.py to reduce coupling.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError


class ModelSelectionService:
    """
    Service for selecting the best model for prediction context.
    
    Responsibilities:
    - Detect available model types from filesystem
    - Filter model candidates
    - Select best model using contextual selectors
    - Resolve model aliases
    """

    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger('ModelSelectionService')

    def get_available_model_types(self) -> set:
        """
        Get available model types by scanning model files in the database directory.
        
        Returns:
            Set of available model type strings
        """
        try:
            base_dir = Path(self.config_manager.get(
                self.ACCUMULATION_OUTPUT_DIR_CONFIG,
                self.DEFAULT_ACCUMULATION_DIR
            ))
            batch_dir = base_dir / 'main_database'
            
            if not batch_dir.exists():
                self.logger.warning(f'Model directory not found: {batch_dir}')
                return {'mlp', 'tabnet'}
            
            model_types = set()
            
            # Check for pickle files (MLP)
            pkl_files = list(batch_dir.glob('*.pkl'))
            if pkl_files:
                model_types.add('mlp')
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'Found {len(pkl_files)} MLP models')
            
            # Check for zip files (TabNet)
            zip_files = list(batch_dir.glob('*.zip'))
            if zip_files:
                model_types.add('tabnet')
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'Found {len(zip_files)} TabNet models')
            
            # Check for Keras files (CNN, LSTM, GRU, Transformer, Autoencoder)
            keras_files = list(batch_dir.glob('*.keras'))
            if keras_files:
                model_types.update(['cnn', 'lstm', 'gru', 'transformer', 'autoencoder'])  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'Found {len(keras_files)} Keras models')
            
            self.logger.info(f'Available model types: {sorted(model_types)}')
            return model_types if model_types else {'mlp', 'tabnet'}
        
        except Exception as e:
            self.logger.error(f'Error scanning model types: {e}', exc_info=True)
            raise DataProcessingError(f"Failed to scan available model types: {e}") from e

    def filter_models_by_type(
        self,
        models_meta: Dict[str, Any],
        available_model_types: set
    ) -> Dict[str, Any]:
        """
        Filter models metadata to only include available model types.
        
        Args:
            models_meta: Full models metadata dictionary
            available_model_types: Set of available model type strings
            
        Returns:
            Filtered models metadata
        """
        filtered_models_meta = {}
        for context_id, meta in models_meta.items():
            model_type = meta.get('model_type', '')
            if model_type in available_model_types:
                filtered_models_meta[context_id] = meta
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Skipping {context_id} - {model_type} models not available'
                    )
        
        self.logger.info(
            f'Filtered to {len(filtered_models_meta)}/{len(models_meta)} available contexts'
        )
        return filtered_models_meta

    def select_best_model_for_context(
        self,
        ticker_df_clean,
        meta: Dict[str, Any],
        models: Dict[str, Any],
        ticker: str,
        market_regime: str,
        context_selector: Union[Any, None],
        diary: Any | None = None,
    ) -> str:
        """
        Select the best model for a given context.
        
        Args:
            ticker_df_clean: Prepared ticker DataFrame
            meta: Model metadata
            models: Available models dictionary
            ticker: Ticker symbol
            market_regime: Market regime string
            context_selector: Model selector instance (SmartModelSelector or AdaptiveModelSelector)
            
        Returns:
            Selected model name or empty string if selection fails
        """
        models_list = self._get_prediction_model_candidates(models)
        if not models_list:
            return ''
        
        target_type = meta.get('target_type', 'classification')
        
        # 0) Diary-based contextual selection (exact or KNN-expanded).
        try:
            if diary is not None:
                context_fingerprint = self._create_context_fingerprint(
                    ticker_df_clean, market_regime
                )
                context_pattern_seq = self._get_current_context_pattern_seq(
                    ticker_df_clean)
                weights = {}
                get_knn = getattr(diary, "get_knn_contextual_model_weights", None)
                if callable(get_knn):
                    weights = get_knn(
                        context_fingerprint,
                        context_pattern_seq=context_pattern_seq,
                    )
                else:
                    get_exact = getattr(diary, "get_contextual_model_weights", None)
                    if callable(get_exact):
                        weights = get_exact(context_fingerprint)

                if weights:
                    # Choose best among available candidates.
                    scored = {
                        m: self._score_model_from_context_weights(m, weights)
                        for m in models_list
                    }
                    best = max(scored, key=scored.get)
                    if scored.get(best, 0.0) > 0:
                        self.logger.info(
                            f"DiarySelector chose '{best}' for {ticker} in {market_regime} regime (fp={context_fingerprint})."
                        )
                        return best
        except Exception as e:
            self.logger.warning(
                f"Diary-based selection failed for {ticker}: {e}", exc_info=True
            )

        # Use adaptive selector if available
        if hasattr(context_selector, 'select_best_model_adaptive'):
            context_fingerprint = self._create_context_fingerprint(ticker_df_clean, market_regime)
            selected_name = context_selector.select_best_model_adaptive(context_fingerprint)
            best_model_name = (
                self._resolve_model_selection(selected_name, models_list) or models_list[0]
            )
        else:
            # Use smart selector
            model_aliases = self._build_model_alias_map(models_list)
            if len(model_aliases) == 1:
                best_model_name = next(iter(model_aliases.values()))
            else:
                selected_type = context_selector.select_best_model(
                    ticker_df_clean, target_type, list(model_aliases.keys())
                )[0]
                best_model_name = (
                    model_aliases.get(selected_type) or
                    self._resolve_model_selection(selected_type, models_list) or
                    models_list[0]
                )
        
        self.logger.info(
            f"Contextual Selector chose '{best_model_name}' for {ticker} in {market_regime} regime."
        )
        return best_model_name or ''

    def _get_prediction_model_candidates(self, models: Dict[str, Any]) -> List[str]:
        """Get list of model names excluding autoencoders."""
        prediction_models = [name for name in models if 'autoencoder' not in name.lower()]  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
        return prediction_models

    def _build_model_alias_map(self, models_list: List[str]) -> Dict[str, str]:
        """Build mapping from model type aliases to actual model names."""
        aliases: Dict[str, str] = {}
        for model_name in models_list:
            aliases.setdefault(self._model_type_alias(model_name), model_name)
        return aliases

    def _resolve_model_selection(self, selected_name: str, models_list: List[str]) -> Optional[str]:
        """Resolve selected model name to actual model name from list."""
        if selected_name in models_list:
            return selected_name
        
        selected_alias = self._model_type_alias(selected_name)
        for model_name in models_list:
            if self._model_type_alias(model_name) == selected_alias:
                return model_name
        return None

    def _score_model_from_context_weights(
        self, model_name: str, weights: Dict[str, float]
    ) -> float:
        """Score a model using direct model ids or model-type aliases."""
        if model_name in weights:
            return float(weights[model_name])
        model_alias = self._model_type_alias(model_name)
        score = 0.0
        for weighted_name, value in weights.items():
            if self._model_type_alias(str(weighted_name)) == model_alias:
                score += float(value)
        return score

    def _model_type_alias(self, model_name: str) -> str:
        """Get canonical alias for a model name."""
        normalized = model_name.lower().replace('-', '_')
        known_aliases = {
            'random_forest': ('random_forest', 'randomforest'),
            'lightgbm': ('lightgbm', 'lgbm'),
            'catboost': ('catboost',),
            'xgboost': ('xgboost', 'xgb'),
            'elasticnet': ('elasticnet', 'elastic_net'),
            'linear': ('linear', 'linear_regression'),
            'ridge': ('ridge',),
            'lstm': ('lstm',),
            'transformer': ('transformer',),
            'mlp': ('mlp',),
            'svm': ('svm',),
            'knn': ('knn',),
        }
        
        for canonical, aliases in known_aliases.items():
            if any(alias in normalized for alias in aliases):
                return canonical
        
        parts = [part for part in normalized.split('_') if part]
        return parts[-1] if parts else normalized

    def _create_context_fingerprint(self, ticker_df, market_regime: str) -> str:
        """Create context fingerprint using context_pattern_id."""
        if 'context_pattern_id' in ticker_df.columns and len(ticker_df) > 0:
            return str(ticker_df['context_pattern_id'].iloc[-1])
        
        # Fallback to legacy logic
        try:
            regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
            regime_val = regime_map.get(market_regime.lower(), 0)
            return f"legacy_{regime_val}"
        except Exception as e:
            self.logger.error(f"Error creating context fingerprint: {e}", exc_info=True)
            return 'unknown_context'

    def _get_current_context_pattern_seq(self, ticker_df) -> Optional[str]:
        if 'context_pattern_seq' in ticker_df.columns and len(ticker_df) > 0:
            value = ticker_df['context_pattern_seq'].iloc[-1]
            return None if value is None else str(value)
        return None
