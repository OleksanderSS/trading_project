from typing import Any


class ModelRegistry:
    """
    Central registry for all models, aliases, and their configurations.
    This serves as the single source of truth for model-related metadata.
    """

    MODELS: dict[str, dict[str, Any]] = {  # audit-ignore: HARDCODED_MODEL_MAP_OR_ALIASES,DUPLICATED_MODEL_REGISTRY_ENTRIES
        'lgbm': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'rf': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'xgboost': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'catboost': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'linear': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'mlp': {'type': 'light', 'role': 'predictor', 'can_be_primary': True},
        'ensemble': {'type': 'light', 'role': 'ensemble', 'can_be_primary': True},

        'lstm': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        'gru': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        'transformer': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        'cnn': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
        'tabnet': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True},
    }

    @classmethod
    def get_models_by_type(cls, model_type: str) -> list[str]:
        return [name for name, cfg in cls.MODELS.items() if cfg['type'] == model_type]

    @classmethod
    def get_all_model_names(cls) -> list[str]:
        return list(cls.MODELS.keys())
