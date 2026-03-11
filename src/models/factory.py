# src/models/factory.py

import importlib
import logging
from typing import Dict, Any, Type, Optional

# The legacy interface is good, let's reuse it for now.
# We will need to move it later as well.
from src.models.interfaces import BaseModel 
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelFactory")

class ModelFactory:
    """
    Central point for obtaining model instances.
    This factory dynamically imports models to avoid crashing if a dependency is missing.
    """

    # Mapping of model identifiers to their import paths
    _MODEL_REGISTRY: Dict[str, str] = {
        'catboost': 'src.models.tree.catboost_model.CatBoostModel',
        'xgboost': 'src.models.tree.xgboost_model.XGBoostModel',
        'cnn': 'src.models.neural.cnn_model.CNNModel',
        'autoencoder': 'src.models.neural.autoencoder_model.AutoencoderModel',
        'ensemble': 'src.models.ensemble.ensemble_model.EnsembleModel',
        'knn': 'src.models.linear.knn_model.KNNModel',
        'linear': 'src.models.linear.linear_model.LinearModel',
        'svm': 'src.models.linear.svm_model.SVMModel',
        'lgbm': 'src.models.tree.lightgbm_model.LightGBMModel',
        'rf': 'src.models.tree.random_forest_model.RandomForestModel',
        'gru': 'src.models.neural.gru_model.GRUModel',
        'lstm': 'src.models.neural.lstm_model.LSTMModel',
        'mlp': 'src.models.neural.mlp_model.MLPModel',
        'transformer': 'src.models.neural.transformer_model.TransformerModel',
    }

    @staticmethod
    def get_model(model_name: str, **kwargs) -> Optional[BaseModel]:
        """
        Dynamically imports, instantiates, and returns a model.

        If the model's dependencies are not installed, it logs a warning
        and returns None.

        Args:
            model_name: The identifier of the model (e.g., 'catboost', 'cnn').
            **kwargs: Configuration parameters for the model constructor.

        Returns:
            An instance of a BaseModel subclass, or None if dependencies are missing.

        Raises:
            ValueError: If the model_name is not found in the registry.
        """
        name_lower = model_name.lower()

        if name_lower not in ModelFactory._MODEL_REGISTRY:
            available = list(ModelFactory._MODEL_REGISTRY.keys())
            error_msg = f"Model '{model_name}' not found. Available models: {available}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        import_path_str = ModelFactory._MODEL_REGISTRY[name_lower]

        try:
            module_path, class_name = import_path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            logger.info(f"Instantiating model: {name_lower} with params: {kwargs}")
            return model_class(**kwargs)

        except ImportError as e:
            logger.warning(
                f"Could not import dependencies for model '{model_name}'. "
                f"Skipping. Please install required libraries if you need this model. Error: {e}"
            )
            return None  # Gracefully fail

        except Exception as e:
            logger.error(f"Failed to instantiate model '{model_name}': {e}")
            raise

    @staticmethod
    def get_available_models() -> list:
        """Returns a list of all registered model identifiers."""
        return list(ModelFactory._MODEL_REGISTRY.keys())
