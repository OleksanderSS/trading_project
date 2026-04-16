import logging
from typing import Dict, Any, Type, Optional, List

from src.config.unified_config_manager import UnifiedConfigManager
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
from src.models.neural.lstm_model import LSTMModel
from src.models.neural.gru_model import GRUModel
from src.models.neural.cnn_model import CNNModel
from src.models.neural.transformer_model import TransformerModel
from src.models.neural.tabnet_model import TabNetModel
from src.models.ensemble.ensemble_model import EnsembleModel

logger = ProjectLogger.get_logger("ModelFactory")

class ModelFactory:
    """
    A factory for creating machine learning models based on configuration.
    It maps model names from the config to their respective classes.
    """
    
    # Mapping of model names (as in config) to their class definitions
    _model_map: Dict[str, Type[BaseModel]] = {
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel,
        "RandomForest": RandomForestModel,
        "Linear": LinearModel,
        "SVM": SVMModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "CNN": CNNModel,
        "Transformer": TransformerModel,
        "TabNet": TabNetModel,
        "Ensemble": EnsembleModel,
    }

    @staticmethod
    def create_model(model_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseModel:
        """
        Creates an instance of a model with the given name and configuration.

        Args:
            model_name (str): The name of the model to create (e.g., 'XGBoost').
            config (Optional[Dict[str, Any]]): The configuration dictionary for the model.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            BaseModel: An instance of the requested model.

        Raises:
            ValueError: If the model_name is not found in the factory's map.
        """
        model_class = ModelFactory._model_map.get(model_name)
        if not model_class:
            logger.error(f"Model '{model_name}' not found in factory.")
            raise ValueError(f"Unsupported model name: {model_name}")

        try:
            logger.info(f"Creating instance of model: {model_name}")
            
            # ✅ СПЕЦІАЛЬНА ОБРОБКА ДЛЯ АНСАМБЛЮ
            if model_name == "Ensemble" and config:
                base_models_names = config.get('models', [])
                if not base_models_names:
                    # Fallback to default if no models specified
                    base_models_names = ["XGBoost", "LightGBM"]
                
                resolved_models = []
                for m_name in base_models_names:
                    m_instance = ModelFactory.create_model(m_name, config=config.get('per_model', {}).get(m_name, {}))
                    resolved_models.append((m_name, m_instance.model))
                
                return EnsembleModel(models=resolved_models, task_type=kwargs.get('task_type', 'classification'))

            return model_class(config=config, **kwargs)
        except Exception as e:
            logger.error(f"Error creating model '{model_name}': {e}", exc_info=True)
            raise

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Returns a list of all available model names.
        """
        return list(ModelFactory._model_map.keys())