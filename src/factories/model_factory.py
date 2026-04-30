import logging
from typing import Dict, Any, Type, Optional, List

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.logging.logger import ProjectLogger

# Updated import path
from src.models.interfaces import BaseModel

# Import all model classes
from src.models.tree.xgboost_model import XGBoostModel
from src.models.tree.lightgbm_model import LightGBMModel
from src.models.tree.catboost_model import CatBoostModel
from src.models.tree.random_forest_model import RandomForestModel
from src.models.linear.linear_model import LinearModel
from src.models.linear.svm_model import SVMModel
from src.models.linear.knn_model import KNNModel
from src.models.neural.lstm_model import LSTMModel
from src.models.neural.gru_model import GRUModel
from src.models.neural.cnn_model import CNNModel
from src.models.neural.transformer_model import TransformerModel
from src.models.neural.tabnet_model import TabNetModel
from src.models.neural.mlp_model import MLPModel
from src.models.neural.autoencoder_model import AutoencoderModel
from src.models.ensemble.ensemble_model import EnsembleModel

logger = ProjectLogger.get_logger("ModelFactory")

class ModelFactory:
    """
    A factory for creating machine learning models based on configuration.
    It maps model names from the config to their respective classes.
    """
    
    # Mapping of canonical model names to their class definitions
    _model_map: Dict[str, Type[BaseModel]] = {
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel,
        "RandomForest": RandomForestModel,
        "Linear": LinearModel,
        "SVM": SVMModel,
        "KNN": KNNModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "CNN": CNNModel,
        "Transformer": TransformerModel,
        "TabNet": TabNetModel,
        "MLP": MLPModel,
        "Autoencoder": AutoencoderModel,
        "Ensemble": EnsembleModel,
    }

    _model_aliases: Dict[str, str] = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "random_forest": "RandomForest",
        "randomforest": "RandomForest",
        "linear": "Linear",
        "svm": "SVM",
        "knn": "KNN",
        "mlp": "MLP",
        "cnn": "CNN",
        "lstm": "LSTM",
        "gru": "GRU",
        "transformer": "Transformer",
        "tabnet": "TabNet",
        "autoencoder": "Autoencoder",
        "ensemble": "Ensemble",
    }

    @staticmethod
    def create_model(model_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseModel:
        """
        Creates an instance of a model with a given name and configuration.

        Args:
            model_name (str): The name of the model to create (e.g., 'XGBoost' or 'xgboost').
            config (Optional[Dict[str, Any]]): The configuration dictionary for the model.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            BaseModel: An instance of the requested model.

        Raises:
            ValueError: If the model_name is not found in the factory's map.
        """
        canonical_name = ModelFactory._validate_and_normalize_name(model_name)
        model_class = ModelFactory._get_model_class(canonical_name, model_name)
        
        logger.info(f"Creating instance of model: {canonical_name}")
        
        # Handle special Ensemble case
        if canonical_name == "Ensemble":
            return ModelFactory._create_ensemble_model(config, kwargs)
        
        # Create regular model with parameters
        return ModelFactory._create_regular_model(model_class, canonical_name, config, kwargs)
    
    @staticmethod
    def _validate_and_normalize_name(model_name: str) -> str:
        """Validate and normalize model name"""
        if not isinstance(model_name, str) or not model_name.strip():
            logger.error("create_model called without a valid model_name")
            raise ValueError("Unsupported model name: {model_name}")
        
        normalized_name = model_name.strip()
        lookup_key = normalized_name.lower().replace('-', '_')
        return ModelFactory._model_aliases.get(lookup_key, normalized_name)
    
    @staticmethod
    def _get_model_class(canonical_name: str, original_name: str):
        """Get model class from canonical name"""
        model_class = ModelFactory._model_map.get(canonical_name)
        if not model_class:
            logger.error(f"Model '{original_name}' not found in factory.")
            raise ValueError(f"Unsupported model name: {original_name}")
        return model_class
    
    @staticmethod
    def _create_ensemble_model(config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> BaseModel:
        """Create ensemble model with base models"""
        base_models_names = config.get('models', []) if config else []
        if not base_models_names:
            # Fallback to default if no models specified
            base_models_names = ["XGBoost", "LightGBM"]
        
        resolved_models = []
        for m_name in base_models_names:
            m_instance = ModelFactory.create_model(m_name, config=config.get('per_model', {}).get(m_name, {}))
            resolved_models.append((m_name, m_instance.model))
        
        return EnsembleModel(models=resolved_models, task_type=kwargs.get('task_type', 'classification'))
    
    @staticmethod
    def _create_regular_model(model_class, canonical_name: str, config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> BaseModel:
        """Create regular model with parameter processing"""
        model_params = ModelFactory._extract_model_params(canonical_name, config)
        all_params = {**model_params, **kwargs}
        
        return ModelFactory._create_model_with_filtered_params(model_class, canonical_name, all_params)
    
    @staticmethod
    def _extract_model_params(canonical_name: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract model-specific parameters from config"""
        model_params = {}
        if config and canonical_name == "KNN" and 'n_neighbors' in config:
            model_params['n_neighbors'] = config['n_neighbors']
        return model_params
    
    @staticmethod
    def _create_model_with_filtered_params(model_class, canonical_name: str, all_params: Dict[str, Any]) -> BaseModel:
        """Create model with parameter filtering and seed injection"""
        import inspect
        
        # Get global seed for reproducibility
        config_manager = get_current_config()
        global_seed = config_manager.get('performance.random_seed', 42)
        
        try:
            sig = inspect.signature(model_class.__init__)
            accepted_params = set(sig.parameters.keys()) - {'self'}
            
            # Add random seed if model supports it
            ModelFactory._add_random_seed_if_supported(all_params, accepted_params, global_seed)
            
            # Filter parameters
            filtered_params = {k: v for k, v in all_params.items() if k in accepted_params}
            logger.debug(f"Filtered params for {canonical_name}: {list(filtered_params.keys())}")
            return model_class(**filtered_params)
        except Exception as inspect_error:
            logger.warning(f"Could not inspect {canonical_name} constructor signature: {inspect_error}. Passing all params.")
            return model_class(**all_params)
    
    @staticmethod
    def _add_random_seed_if_supported(all_params: Dict[str, Any], accepted_params: set, global_seed: int) -> None:
        """Add random seed to parameters if model supports it"""
        if 'random_state' in accepted_params and 'random_state' not in all_params:
            all_params['random_state'] = global_seed
        elif 'seed' in accepted_params and 'seed' not in all_params:
            all_params['seed'] = global_seed

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Returns a list of all available model names.
        """
        return list(ModelFactory._model_map.keys())
