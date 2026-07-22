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
from typing import Any, ClassVar

from src.core.logging.logger import ProjectLogger
from src.models import constants


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

    # Known multi-word model type suffixes (longest-first so most specific wins).
    _KNOWN_MODEL_TYPES: ClassVar[list[str]] = [
        'random_forest', 'neural_network', 'base_neural',
        'lightgbm', 'catboost', 'xgboost', 'tabnet',
        'autoencoder', 'transformer', 'lstm', 'gru', 'cnn',
        'mlp', 'linear', 'ensemble', 'knn', 'svm',
    ]

    def get_available_model_types(self) -> set:
        """
        Get available model types by combining:
        1. ModelRegistry registered model names/types
        2. Actual model files (.pkl, .keras, .joblib) found in the batch directory

        Returns:
            Set of available model type strings
        """
        from src.models.registry.model_registry import ModelRegistry

        # 1. Collect types from ModelRegistry
        model_types: set = set()
        available_models = ModelRegistry.get_all_model_names()
        for name in available_models:
            config = ModelRegistry.get_model_config(name)
            if config:
                model_types.add(config.get('type', 'light'))
            # Also treat the model name itself as an available type
            # so that metadata entries whose model_type matches the name pass through
            model_types.add(name.lower())

        # 2. Scan the batch directory for actual model files and infer types
        batch_model_types = self._infer_model_types_from_batch_dir()
        model_types.update(batch_model_types)

        self.logger.info(f'Available model types (registry + batch scan): {sorted(model_types)}')
        return model_types if model_types else {constants.MLP, constants.TABNET}

    def _infer_model_types_from_batch_dir(self) -> set:
        """
        Scan the accumulated batch directory for model files and infer their types
        from filenames.  Expected filename pattern:
            model_<TICKER>_<TARGET>_<MODEL_TYPE>.pkl  (or .keras / .joblib)
        """
        from pathlib import Path

        model_types: set = set()
        model_extensions = {'.pkl', '.keras', '.h5', '.pt', '.joblib'}

        try:
            base_dir = Path(
                self.config_manager.get(
                    self.ACCUMULATION_OUTPUT_DIR_CONFIG,
                    self.DEFAULT_ACCUMULATION_DIR,
                )
            )

            # Collect candidate directories: the base dir itself plus any
            # immediate subdirectories (one batch level deep).
            candidate_dirs: list[Path] = []
            if base_dir.exists():
                candidate_dirs.append(base_dir)
                for entry in base_dir.iterdir():
                    if entry.is_dir():
                        candidate_dirs.append(entry)
                        models_subdir = entry / 'models'
                        if models_subdir.is_dir():
                            candidate_dirs.append(models_subdir)

            for search_dir in candidate_dirs:
                for f in search_dir.iterdir():
                    if not f.is_file() or f.suffix.lower() not in model_extensions:
                        continue
                    inferred = self._infer_type_from_stem(f.stem)
                    if inferred:
                        model_types.add(inferred)

        except Exception as e:
            self.logger.warning(f'Could not scan batch dir for model types: {e}')

        if model_types:
            self.logger.info(f'Inferred model types from batch files: {sorted(model_types)}')
        return model_types

    def _infer_type_from_stem(self, stem: str) -> str | None:
        """
        Infer the model type from a filename stem such as
        'model_AMD_target_up_1d_random_forest'.
        Returns the model type string or None if it cannot be determined.
        """
        s = stem.lower()
        if s.startswith('model_'):
            s = s[len('model_'):]

        for mt in self._KNOWN_MODEL_TYPES:
            if s.endswith('_' + mt):
                return mt
        return None

    def filter_models_by_type(
        self,
        models_meta: dict[str, Any],
        available_model_types: set
    ) -> dict[str, Any]:
        """
        Filter models metadata to only include available model types.

        Args:
            models_meta: Full models metadata dictionary
            available_model_types: Set of available model type strings

        Returns:
            Filtered models metadata
        """
        from src.models.registry.model_registry import ModelRegistry

        filtered_models_meta = {}
        for context_id, meta in models_meta.items():
            # Resolve model type using the registry
            model_name = meta.get('model_type', '')
            config = ModelRegistry.get_model_config(model_name.lower())
            model_type = config.get('type', 'light') if config else 'light'

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
        meta: dict[str, Any],
        models: dict[str, Any],
        ticker: str,
        market_regime: str,
        context_selector: Any | None,
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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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

    def _get_prediction_model_candidates(self, models: dict[str, Any]) -> list[str]:
        """Get list of model names excluding autoencoders."""
        prediction_models = [name for name in models if 'autoencoder' not in name.lower()]  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
        return prediction_models

    def _build_model_alias_map(self, models_list: list[str]) -> dict[str, str]:
        """Build mapping from model type aliases to actual model names using ModelRegistry."""
        from src.models.registry.model_registry import ModelRegistry

        aliases: dict[str, str] = {}
        for model_name in models_list:
            # Simplification: use the registry-defined class or name
            config = ModelRegistry.get_model_config(model_name.lower())
            alias = config.get('class', model_name).lower() if config else model_name
            aliases.setdefault(alias, model_name)
        return aliases

    def _resolve_model_selection(self, selected_name: str, models_list: list[str]) -> str | None:
        """Resolve selected model name to actual model name from list using ModelRegistry."""
        from src.models.registry.model_registry import ModelRegistry

        if selected_name in models_list:
            return selected_name

        # Resolve alias
        for model_name in models_list:
            config = ModelRegistry.get_model_config(model_name.lower())
            if config and config.get('class', '').lower() == selected_name.lower():
                return model_name
        return None

    def _score_model_from_context_weights(
        self, model_name: str, weights: dict[str, float]
    ) -> float:
        """Score a model using direct model ids or model-type aliases."""
        from src.models.registry.model_registry import ModelRegistry

        if model_name in weights:
            return float(weights[model_name])

        config = ModelRegistry.get_model_config(model_name.lower())
        model_alias = config.get('class', model_name).lower() if config else model_name

        score = 0.0
        for weighted_name, value in weights.items():
            w_config = ModelRegistry.get_model_config(str(weighted_name).lower())
            w_alias = w_config.get('class', weighted_name).lower() if w_config else weighted_name

            if w_alias == model_alias:
                score += float(value)
        return score


    def _create_context_fingerprint(self, ticker_df, market_regime: str) -> str:
        """Create context fingerprint using context_pattern_id."""
        if 'context_pattern_id' in ticker_df.columns and len(ticker_df) > 0:
            return str(ticker_df['context_pattern_id'].iloc[-1])

        # Fallback to legacy logic
        try:
            regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
            regime_val = regime_map.get(market_regime.lower(), 0)
            return f"legacy_{regime_val}"
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error creating context fingerprint: {e}", exc_info=True)
            return 'unknown_context'

    def _get_current_context_pattern_seq(self, ticker_df) -> str | None:
        if 'context_pattern_seq' in ticker_df.columns and len(ticker_df) > 0:
            value = ticker_df['context_pattern_seq'].iloc[-1]
            return None if value is None else str(value)
        return None
