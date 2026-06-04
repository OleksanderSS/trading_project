import logging
from typing import Any

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger

# Нові фабрики
from src.factories.tree_model_factory import TreeModelFactory
from src.models.ensemble.ensemble_model import EnsembleModel
from src.models.interfaces import BaseModel
from src.models.linear.knn_model import KNNModel

# Моделі, що залишилися (поки що)
from src.models.linear.linear_model import LinearModel
from src.models.linear.svm_model import SVMModel
from src.models.neural.autoencoder_model import AutoencoderModel  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
from src.models.neural.cnn_model import CNNModel
from src.models.neural.gru_model import GRUModel
from src.models.neural.lstm_model import LSTMModel
from src.models.neural.mlp_model import MLPModel
from src.models.neural.tabnet_model import TabNetModel
from src.models.neural.transformer_model import TransformerModel

logger = ProjectLogger.get_logger('ModelFactory')


class ModelFactory:
    """
    A factory for creating machine learning models based on configuration.
    It maps model names from the config to their respective classes.
    """
    _model_map: dict[str, Any] = {
        'Linear': LinearModel, 'SVM': SVMModel, 'KNN': KNNModel,
        'LSTM': LSTMModel, 'GRU': GRUModel, 'CNN': CNNModel,
        'Transformer': TransformerModel, 'TabNet': TabNetModel, 'MLP':
        MLPModel, 'Autoencoder': AutoencoderModel, 'Ensemble': EnsembleModel  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
    }
    _model_aliases: dict[str, str] = {
        'linear': 'Linear', 'svm': 'SVM', 'knn': 'KNN',
        'mlp': 'MLP', 'cnn': 'CNN', 'lstm': 'LSTM', 'gru': 'GRU',
        'transformer': 'Transformer', 'tabnet': 'TabNet', 'autoencoder':  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
        'Autoencoder', 'ensemble': 'Ensemble', 'xgboost': 'XGBoost',
        'xgb': 'XGBoost', 'lightgbm': 'LightGBM', 'lgbm': 'LightGBM',
        'catboost': 'CatBoost', 'random_forest': 'RandomForest',
        'randomforest': 'RandomForest', 'rf': 'RandomForest'
    }

    @staticmethod
    def create_model(model_name: str, config: dict[str, Any] | None=None,
        **kwargs) ->BaseModel:
        """Creates an instance of a model."""
        canonical_name = ModelFactory._validate_and_normalize_name(model_name)

        # Делегуємо деревні моделі
        if TreeModelFactory.is_tree_model(canonical_name):
            return TreeModelFactory.create_model(canonical_name, config=config, **kwargs)

        # Інші моделі
        model_class = ModelFactory._get_model_class(canonical_name, model_name)
        logger.info(f'Creating instance of model: {canonical_name}')

        if canonical_name == 'Ensemble':
            return ModelFactory._create_ensemble_model(config, kwargs)

        return ModelFactory._create_regular_model(model_class,
            canonical_name, config, kwargs)

    @staticmethod
    def _validate_and_normalize_name(model_name: str) ->str:
        """Validate and normalize model name"""
        if not isinstance(model_name, str) or not model_name.strip():
            logger.error('create_model called without a valid model_name')
            raise ValueError('Unsupported model name: {model_name}')
        normalized_name = model_name.strip()
        lookup_key = normalized_name.lower().replace('-', '_')
        return ModelFactory._model_aliases.get(lookup_key, normalized_name)

    @staticmethod
    def _get_model_class(canonical_name: str, original_name: str):
        """Get model class from canonical name"""
        model_class = ModelFactory._model_map.get(canonical_name)
        if not model_class:
            logger.error(f"Model '{original_name}' not found in factory.")
            raise ValueError(f'Unsupported model name: {original_name}')
        return model_class

    @staticmethod
    def _create_ensemble_model(config: dict[str, Any] | None, kwargs:
        dict[str, Any]) ->BaseModel:
        """Create ensemble model with base models"""
        base_models_names = config.get('models', []) if config else []
        if not base_models_names:
            base_models_names = ['XGBoost', 'LightGBM']
        resolved_models = []
        for m_name in base_models_names:
            per_model_config: dict[str, Any] = {}
            if config:
                per_model_config = config.get('per_model', {}).get(m_name, {})
            m_instance = ModelFactory.create_model(m_name, config=
                per_model_config)
            if hasattr(m_instance, 'model') and m_instance.model is not None:
                resolved_models.append((m_name, m_instance.model))
            else:
                resolved_models.append((m_name, m_instance))
        return EnsembleModel(models=resolved_models, task_type=kwargs.get(
            'task_type', 'classification'))

    @staticmethod
    def _create_regular_model(model_class, canonical_name: str, config:
        dict[str, Any] | None, kwargs: dict[str, Any]) ->BaseModel:
        """Create regular model with parameter processing"""
        model_params = ModelFactory._extract_model_params(canonical_name,
            config)
        all_params = {**model_params, **kwargs}
        return ModelFactory._create_model_with_filtered_params(model_class,
            canonical_name, all_params)

    @staticmethod
    def _extract_model_params(canonical_name: str, config: dict[str, Any] | None) ->dict[str, Any]:
        """Extract model-specific parameters from config"""
        model_params = {}
        if config and canonical_name == 'KNN' and 'n_neighbors' in config:
            model_params['n_neighbors'] = config['n_neighbors']
        return model_params

    @staticmethod
    def _create_model_with_filtered_params(model_class: type[BaseModel],
        canonical_name: str, all_params: dict[str, Any]) ->BaseModel:
        """Create model with parameter filtering and seed injection"""
        import inspect
        config_manager = get_current_config()
        global_seed = config_manager.get('performance.random_seed', 42)
        try:
            sig = inspect.signature(model_class.__init__)
            accepted_params = set(sig.parameters.keys()) - {'self'}
            ModelFactory._add_random_seed_if_supported(all_params,
                accepted_params, global_seed)
            filtered_params = {k: v for k, v in all_params.items() if k in
                accepted_params}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'Filtered params for {canonical_name}: {list(filtered_params.keys())}'
                    )
            model_instance: BaseModel = model_class(**filtered_params)
            return model_instance
        except Exception as inspect_error:
            logger.error(f'Виникла помилка: {inspect_error}', exc_info
                =True)
            logger.warning(
                f'Could not inspect {canonical_name} constructor signature: {inspect_error}. Passing all params.'
                )
            model_instance_fallback: BaseModel = model_class(**all_params)
            return model_instance_fallback

    @staticmethod
    def _add_random_seed_if_supported(all_params: dict[str, Any],
        accepted_params: set, global_seed: int) ->None:
        """Add random seed to parameters if model supports it"""
        if ('random_state' in accepted_params and 'random_state' not in
            all_params):
            all_params['random_state'] = global_seed
        elif 'seed' in accepted_params and 'seed' not in all_params:
            all_params['seed'] = global_seed

    @staticmethod
    def get_available_models() ->list[str]:
        """
        Returns a list of all available model names.
        """
        return list(ModelFactory._model_map.keys())

    @staticmethod
    def get_model(model_name: str, **kwargs) ->BaseModel | None:
        """
        Legacy API: Get model instance with kwargs (backward compatible with src/models/factory.py).

        This method provides backward compatibility for code that uses get_model() instead of create_model().
        It dynamically handles missing dependencies and returns None gracefully.

        Args:
            model_name: The identifier of the model (e.g., 'catboost', 'cnn', 'xgboost').
            **kwargs: Configuration parameters for the model constructor.

        Returns:
            An instance of a BaseModel subclass, or None if dependencies are missing.

        Raises:
            ValueError: If the model_name is not found in the factory's map.

        Example:
            model = ModelFactory.get_model('catboost', iterations=100)
            model = ModelFactory.get_model('lstm', hidden_size=128)
        """
        try:
            return ModelFactory.create_model(model_name, config=None, **kwargs)
        except ImportError as e:
            logger.warning(
                f"Could not import dependencies for model '{model_name}'. Skipping. Please install required libraries if you need this model. Error: {e}"
                )
            return None
        except Exception as e:
            logger.error(f"Failed to instantiate model '{model_name}': {e}")
            raise
