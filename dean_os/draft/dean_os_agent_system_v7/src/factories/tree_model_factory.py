import inspect
from typing import Any, ClassVar

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel
from src.models.tree.catboost_model import CatBoostModel
from src.models.tree.lightgbm_model import LightGBMModel
from src.models.tree.random_forest_model import RandomForestModel
from src.models.tree.xgboost_model import XGBoostModel

logger = ProjectLogger.get_logger('TreeModelFactory')

class TreeModelFactory:
    """Factory for creating tree-based models with parameter filtering."""

    _model_map: ClassVar[dict[str, type[BaseModel]]] = {
        'XGBoost': XGBoostModel,
        'LightGBM': LightGBMModel,
        'CatBoost': CatBoostModel,
        'RandomForest': RandomForestModel
    }

    @staticmethod
    def create_model(model_name: str, config: dict[str, Any] | None = None, **kwargs) -> BaseModel:
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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Could not inspect {model_name} constructor: {e}", exc_info=True)
            raise RuntimeError(f"Could not inspect {model_name} constructor: {e}") from e

    @staticmethod
    def is_tree_model(model_name: str) -> bool:
        return model_name in TreeModelFactory._model_map
