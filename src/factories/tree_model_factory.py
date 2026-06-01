import inspect
from typing import Dict, Any, Type, Optional
from src.models.interfaces import BaseModel
from src.models.tree.xgboost_model import XGBoostModel
from src.models.tree.lightgbm_model import LightGBMModel
from src.models.tree.catboost_model import CatBoostModel
from src.models.tree.random_forest_model import RandomForestModel
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('TreeModelFactory')

class TreeModelFactory:
    """Factory for creating tree-based models with parameter filtering."""
    
    _model_map: Dict[str, Type[BaseModel]] = {
        'XGBoost': XGBoostModel,
        'LightGBM': LightGBMModel,
        'CatBoost': CatBoostModel,
        'RandomForest': RandomForestModel
    }

    @staticmethod
    def create_model(model_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseModel:
        model_class = TreeModelFactory._model_map.get(model_name)
        if not model_class:
            raise ValueError(f"Tree model '{model_name}' not supported.")
        
        all_params = {**(config or {}), **kwargs}
        
        # Фільтрація параметрів
        try:
            sig = inspect.signature(model_class.__init__)
            accepted_params = set(sig.parameters.keys()) - {'self'}
            filtered_params = {k: v for k, v in all_params.items() if k in accepted_params}
            
            return model_class(**filtered_params)
        except Exception as e:
            logger.error(f"Could not inspect {model_name} constructor: {e}", exc_info=True)
            raise RuntimeError(f"Could not inspect {model_name} constructor: {e}") from e

    @staticmethod
    def is_tree_model(model_name: str) -> bool:
        return model_name in TreeModelFactory._model_map
