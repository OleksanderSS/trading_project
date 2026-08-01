#!/usr/bin/env python3
"""
Model Registry - Centralized Model Management
Handles model registration, metadata storage, and retrieval.
"""

from datetime import datetime
from typing import Any, ClassVar

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelRegistry")


class ModelRegistry:
    """
    Centralized registry for model management.

    Handles:
    - Model configuration and metadata (Source of Truth)
    - Model registration and storage
    - Model retrieval
    """

    # Unified model configuration map
    MODELS: ClassVar[dict[str, dict[str, Any]]] = {
        'lgbm': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'LightGBM', 'default_feature_count': 64},
        'rf': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'RandomForest', 'default_feature_count': 42},
        'xgboost': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'XGBoost', 'default_feature_count': 64},
        'catboost': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'CatBoost', 'default_feature_count': 64},
        'linear': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'Linear', 'default_feature_count': 20},
        'mlp': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'MLP', 'default_feature_count': 42},
        'ensemble': {'type': 'light', 'role': 'ensemble', 'can_be_primary': True, 'class': 'Ensemble', 'default_feature_count': 42},
        'svm': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'SVM', 'default_feature_count': 30},
        'knn': {'type': 'light', 'role': 'predictor', 'can_be_primary': True, 'class': 'KNN', 'default_feature_count': 20},

        'lstm': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True, 'class': 'LSTM', 'default_feature_count': 42},
        'gru': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True, 'class': 'GRU', 'default_feature_count': 42},
        'transformer': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True, 'class': 'Transformer', 'default_feature_count': 42},
        'cnn': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True, 'class': 'CNN', 'default_feature_count': 42},
        'tabnet': {'type': 'heavy', 'role': 'predictor', 'can_be_primary': True, 'class': 'TabNet', 'default_feature_count': 42},
        'autoencoder': {'type': 'heavy', 'role': 'anomaly', 'can_be_primary': False, 'class': 'Autoencoder', 'default_feature_count': 42},

        # Enhanced/Experimental
        'lgbm_bayesian': {'type': 'enhanced', 'role': 'predictor', 'can_be_primary': True, 'class': 'LightGBM', 'default_feature_count': 64},

        # Aliases/Legacy
        'lightgbm': {'alias_for': 'lgbm'},
        'random_forest': {'alias_for': 'rf'},
    }

    def __init__(self, storage_path: Any = None):
        self.logger = logger
        self._registered_models: dict[str, Any] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self.storage_path = storage_path
        self.logger.info(f"✅ Unified ModelRegistry initialized (storage: {storage_path})")

    @classmethod
    def get_model_config(cls, model_name: str) -> dict[str, Any] | None:
        cfg = cls.MODELS.get(model_name)
        if cfg and 'alias_for' in cfg:
            return cls.MODELS.get(cfg['alias_for'])
        return cfg

    @classmethod
    def get_models_by_type(cls, model_type: str) -> list[str]:
        return [name for name, cfg in cls.MODELS.items() if cfg.get('type') == model_type]

    @classmethod
    def resolve_model_name(cls, model_name: str) -> str:
        """Canonical name for a model, following an alias if one is given.

        The two spellings are both live: MODELS calls them 'lgbm' and 'rf',
        while the models that actually trained are recorded in
        experience_diary as 'lightgbm' and 'random_forest' -- the aliases.
        Anything matching stored artifacts against registry names has to
        canonicalise first.
        """
        config = cls.MODELS.get(model_name)
        if config and 'alias_for' in config:
            return str(config['alias_for'])
        return model_name

    @classmethod
    def get_all_model_names(cls, include_aliases: bool = False) -> list[str]:
        """Every distinct model, once.

        This used to return MODELS.keys() outright, which counts 'lgbm' and
        'lightgbm' -- one model -- as two. battle_groups feeds the result
        straight into BATTLE_GROUPS, so the same model entered a tournament
        twice and drew twice the battle slots. Latent rather than live: the
        arena is not constructed anywhere outside its own module.

        It also disagreed with get_models_by_type, which skips aliases
        because they carry no 'type' -- 18 names one way, 16 the other.
        """
        if include_aliases:
            return list(cls.MODELS.keys())
        return [name for name, cfg in cls.MODELS.items() if 'alias_for' not in cfg]

    def register_model(self, model: Any, model_name: str) -> None:
        """Register model in the registry."""
        if model_name not in self.MODELS:
            self.logger.warning(f"Registering model not in registry: {model_name}")
        self._registered_models[model_name] = model

        # Initialize default metadata if not exists
        if model_name not in self._metadata:
            self._metadata[model_name] = {
                'registration_time': datetime.now(),
                'analysis_count': 0,
                'last_analysis': None
            }

        self.logger.info(f"✅ Model registered: {model_name}")

    def get_model(self, model_name: str) -> Any | None:
        """Get model by name."""
        return self._registered_models.get(model_name)

    def get_model_metadata(self, model_name: str) -> dict[str, Any] | None:
        """Get metadata for a model."""
        return self._metadata.get(model_name)

    def update_metadata(self, model_name: str, metadata: dict[str, Any]) -> None:
        """Update metadata for a model."""
        self._metadata[model_name] = metadata
