#!/usr/bin/env python3
"""
Model Registry - Centralized Model Management
Handles model registration, metadata storage, and retrieval.
"""

from typing import Any

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
    MODELS: dict[str, dict[str, Any]] = {
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
        'dean_ensemble': {'type': 'enhanced', 'role': 'ensemble', 'can_be_primary': True, 'class': 'DeanEnsemble', 'default_feature_count': 64},
        'sentiment': {'type': 'enhanced', 'role': 'predictor', 'can_be_primary': False, 'class': 'SentimentModel', 'default_feature_count': 10},
        'lgbm_bayesian': {'type': 'enhanced', 'role': 'predictor', 'can_be_primary': True, 'class': 'LightGBM', 'default_feature_count': 64},

        # Aliases/Legacy
        'lightgbm': {'alias_for': 'lgbm'},
        'random_forest': {'alias_for': 'rf'},
    }

    def __init__(self):
        self.logger = logger
        self.models: dict[str, Any] = {}
        self.logger.info("✅ Unified ModelRegistry initialized")

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
    def get_all_model_names(cls) -> list[str]:
        return list(cls.MODELS.keys())

    def register_model(self, model: Any, model_name: str) -> None:
        """Register model in the registry."""
        if model_name not in self.MODELS:
            self.logger.warning(f"Registering model not in registry: {model_name}")
        self.models[model_name] = model
        self.logger.info(f"✅ Model registered: {model_name}")

    def get_model(self, model_name: str) -> Any | None:
        """Get model by name."""
        return self.models.get(model_name)
