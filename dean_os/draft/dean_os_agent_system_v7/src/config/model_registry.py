from typing import Any, ClassVar

from src.models import constants


class ModelRegistry:
    """
    Central registry for all models, aliases, and their configurations.
    This serves as the single source of truth for model-related metadata.
    """

    MODELS: ClassVar[dict[str, dict[str, Any]]] = {  # audit-ignore: HARDCODED_MODEL_MAP_OR_ALIASES,DUPLICATED_MODEL_REGISTRY_ENTRIES
        'lgbm': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'rf': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        constants.XGBOOST: {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        constants.CATBOOST: {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'linear': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        constants.MLP: {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'ensemble': {'type': 'light', 'role': 'ensemble', 'can_be_primary': True},

        constants.LSTM: {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        constants.GRU: {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        constants.TRANSFORMER: {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        constants.CNN: {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        constants.TABNET: {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
    }

    @classmethod
    def get_models_by_type(cls, model_type: str) -> list[str]:
        return [name for name, cfg in cls.MODELS.items() if cfg['type'] == model_type]

    @classmethod
    def get_all_model_names(cls) -> list[str]:
        return list(cls.MODELS.keys())
