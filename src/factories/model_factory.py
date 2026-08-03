import logging
from typing import Any, ClassVar

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
from src.models.registry.model_registry import ModelRegistry

logger = ProjectLogger.get_logger('ModelFactory')


class ModelFactory:
    """
    A factory for creating machine learning models based on configuration.
    It maps model names from the config to their respective classes using ModelRegistry.
    """

    # Neural models are resolved on first use, not imported at module load.
    # Importing them eagerly drags TensorFlow/PyTorch-scale dependencies into
    # every process that so much as touches ModelFactory -- including CLI
    # tools and tests that never build a neural model. Tree models were
    # already lazy (their entries below are plain strings delegated to
    # TreeModelFactory); this gives the neural ones the same treatment.
    #
    # The seven neural entries were REMOVED on 2026-08-02 and their classes
    # archived to src/archive/models_neural_unreachable/. They were
    # unreachable from both training paths and had been for some time:
    #
    #   - Locally, _select_models_for_ticker returns models.yaml's `light`
    #     category, which contains none of them.
    #   - In Colab, colab_clean_cell.py builds every model directly with
    #     sklearn MLPRegressor, tf.keras.Sequential and pytorch_tabnet; it
    #     imports none of these classes.
    #   - Prediction does not revive them either: src/models/loader.py loads
    #     saved artifacts with joblib.load and keras load_model on file
    #     paths, not by reconstructing project classes. There were zero
    #     saved .pkl/.keras files at the time of removal, so no artifact
    #     depended on the class definitions.
    #
    # get_available_models() therefore listed 16 model types when 9 could
    # actually be built -- a list consulted by DEFAULT_ENABLED_MODEL_TYPES
    # and handed to ContextualModelSelector as its universe of candidates.
    #
    # src/models/neural/sequence_builder.py is deliberately NOT archived:
    # loader.py uses SequenceBuilder to shape inputs for Colab-trained
    # sequence models at prediction time.
    _lazy_class_paths: ClassVar[dict[str, str]] = {}
    _resolved_classes: ClassVar[dict[str, Any]] = {}

    #: Trained in Colab, never locally. Named here so asking for one gets an
    #: explanation instead of a bare "unknown model".
    _COLAB_ONLY_MODELS: ClassVar[frozenset[str]] = frozenset({
        'lstm', 'gru', 'cnn', 'transformer', 'tabnet', 'mlp', 'autoencoder',
    })

    # Mapping to actual classes
    _class_map: ClassVar[dict[str, Any]] = {
        'Linear': LinearModel, 'SVM': SVMModel, 'KNN': KNNModel,
        'Ensemble': EnsembleModel,
        'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM', 'CatBoost': 'CatBoost',
        'RandomForest': 'RandomForest'
    }

    @staticmethod
    def create_model(model_name: str, config: dict[str, Any] | None=None,
        **kwargs) ->BaseModel:
        """Creates an instance of a model."""
        canonical_name = ModelFactory._validate_and_normalize_name(model_name)

        # Guard: Autoencoder should not be used as primary predictor
        model_config = ModelRegistry.get_model_config(canonical_name.lower())
        if model_config and model_config.get('role') == 'anomaly' and model_config.get('can_be_primary') == False:
            if kwargs.get('use_as_primary', False):
                logger.error(f"Model '{canonical_name}' is an anomaly detection model and cannot be used as primary predictor")
                raise ValueError(f"Model '{canonical_name}' is an anomaly detection model and cannot be used as primary predictor")

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
        """Validate and normalize model name using ModelRegistry"""
        if not isinstance(model_name, str) or not model_name.strip():
            logger.error('create_model called without a valid model_name')
            raise ValueError('Unsupported model name: {model_name}')

        normalized_name = model_name.strip().lower()
        # Direct lookup in Registry
        config = ModelRegistry.get_model_config(normalized_name)
        if config:
            return config.get('class', normalized_name.capitalize())

        # Fallback for aliases/legacy names
        # ... logic to handle aliases if not directly in registry ...
        return normalized_name.capitalize()

    @staticmethod
    def _get_model_class(canonical_name: str, original_name: str):
        """Get model class from canonical name, importing neural models lazily."""
        model_class = ModelFactory._class_map.get(canonical_name)
        if model_class:
            return model_class

        path = ModelFactory._lazy_class_paths.get(canonical_name)
        if path:
            cached = ModelFactory._resolved_classes.get(canonical_name)
            if cached is not None:
                return cached
            import importlib
            module_name, _, class_name = path.partition(':')
            try:
                module = importlib.import_module(module_name)
                resolved = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Could not load model '{original_name}' from {path}: {e}")
                raise ValueError(f"Unsupported model name: {original_name}") from e
            ModelFactory._resolved_classes[canonical_name] = resolved
            return resolved

        if str(original_name).strip().lower() in ModelFactory._COLAB_ONLY_MODELS:
            # Distinguishable from a typo. models.yaml still declares these
            # under categories.heavy -- correctly, since that is the list of
            # what Colab trains -- so a caller can reach here holding a
            # perfectly valid model name that simply is not built on this
            # side of the hybrid split.
            logger.error(
                "Model '%s' is trained in Colab, not locally: it is in "
                "models.yaml categories.heavy and is built inside "
                "scripts/colab/colab_clean_cell.py. The local factory has no "
                "class for it.",
                original_name,
            )
            raise ValueError(
                f"Model '{original_name}' is a Colab-side model and cannot be "
                f"built locally. Local training uses models.yaml "
                f"categories.light."
            )

        logger.error(f"Model '{original_name}' not found in factory.")
        raise ValueError(f'Unsupported model name: {original_name}')

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
        return dict(config) if config else {}

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
        except (TypeError, ValueError, AttributeError, KeyError) as inspect_error:
            logger.error(f'Виникла помилка: {inspect_error}', exc_info=True)
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
        """Model names this factory can actually build.

        ModelRegistry describes every model the SYSTEM knows about, which
        includes the seven trained in Colab. This returned that full list,
        and its callers read it as "what can be trained here":
        DEFAULT_ENABLED_MODEL_TYPES, UnifiedTrainingManager's fallback when
        no category is configured, and ContextualModelSelector's universe of
        candidates -- so the selector could rank, and recommend, a model no
        local code path can construct.

        The registry is still the source; this only removes what this side of
        the hybrid split cannot build. Use ModelRegistry.get_all_model_names()
        directly when the question really is "what models exist".
        """
        return [
            name for name in ModelRegistry.get_all_model_names()
            if str(name).strip().lower() not in ModelFactory._COLAB_ONLY_MODELS
        ]

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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to instantiate model '{model_name}': {e}")
            raise
