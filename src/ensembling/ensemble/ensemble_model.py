# src/models/ensemble/ensemble_model.py

import numpy as np
import joblib
import os
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
from typing import List, Tuple, Any, Optional, Type, Dict
import logging
import importlib

from src.config.unified_config_manager import get_current_config
from src.models.interfaces import BaseModel

logger = logging.getLogger(__name__)

def _import_model_class(class_path: str) -> Type[BaseModel]:
    """Dynamically imports a model class from a given string path."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import model class from path: {class_path}. Error: {e}")
        raise

class EnsembleModel(BaseModel):
    """
    An ensemble model that combines multiple models using a voting strategy.
    This model adheres to the BaseModel interface.
    """
    
    def __init__(self, models: Optional[List[Tuple[str, BaseModel]]] = None, task_type: str = "classification", voting: str = "soft", **kwargs):
        super().__init__(model_type="ensemble", task_type=task_type)
        self.voting = voting
        self.models = models if models is not None else self._get_default_models()
        self.ensemble = self._create_ensemble()

    def _get_default_models(self) -> List[Tuple[str, BaseModel]]:
        """Loads the default list of models from the configuration file."""
        config_manager = get_current_config()
        model_config = config_manager.get_config('models')
        ensemble_config = model_config.get('ensemble', {})
        
        model_set_key = 'classification_models' if self.task_type == "classification" else 'regression_models'
        models_to_load = ensemble_config.get(model_set_key, {})
        
        default_models = []
        for name, config in models_to_load.items():
            try:
                if 'class' not in config:
                    logger.warning(f"Configuration for model '{name}' is missing a 'class' path. Skipping.")
                    continue
                model_class = _import_model_class(config['class'])
                params = config.get('params', {})
                params['task_type'] = self.task_type
                model_instance = model_class(**params)
                default_models.append((name, model_instance))
            except Exception as e:
                logger.warning(f"Could not load model '{name}' for ensemble. Skipping. Error: {e}")
        
        logger.info(f"Loaded {len(default_models)} default models for {self.task_type} ensemble.")
        return default_models
    
    def _create_ensemble(self) -> Any:
        """Creates the VotingClassifier or VotingRegressor instance."""
        if not self._validate_models_exist():
            return None

        logger.info(f"Creating a {self.task_type} ensemble with {len(self.models)} models: {[name for name, _ in self.models]}")

        if not self._filter_compatible_models():
            return None

        return self._create_voting_ensemble()

    def _validate_models_exist(self) -> bool:
        """Validate that models exist for ensemble creation."""
        if not self.models:
            logger.warning("No models provided or loaded for the ensemble. The ensemble will be empty.")
            return False
        return True

    def _filter_compatible_models(self) -> bool:
        """Filter models by task type compatibility."""
        compatible_models = []
        for name, model in self.models:
            if self._is_model_compatible(name, model):
                compatible_models.append((name, model))

        self.models = compatible_models
        if not self.models:
            logger.error("No compatible models found for the ensemble after filtering by task type.")
            return False
        return True

    def _is_model_compatible(self, name: str, model: BaseModel) -> bool:
        """Check if model is compatible with ensemble task type."""
        if hasattr(model, 'task_type') and model.task_type == self.task_type:
            return True
        
        logger.warning(f"Model '{name}' is incompatible with ensemble task_type '{self.task_type}'. Excluding it.")
        return False

    def _create_voting_ensemble(self) -> Any:
        """Create the appropriate voting ensemble based on task type."""
        try:
            if self.task_type == "classification":
                return self._create_classification_ensemble()
            elif self.task_type == "regression":
                return VotingRegressor(estimators=self.models)
            else:
                raise ValueError(f"Unsupported task type for ensemble: {self.task_type}")
        except Exception as e:
            logger.error(f"Failed to create ensemble: {e}", exc_info=True)
            return None

    def _create_classification_ensemble(self) -> VotingClassifier:
        """Create classification ensemble with appropriate voting strategy."""
        if self.voting == 'soft':
            self._check_predict_proba_support()
        
        return VotingClassifier(estimators=self.models, voting=self.voting)

    def _check_predict_proba_support(self):
        """Check if all models support predict_proba for soft voting."""
        for name, model in self.models:
            if not hasattr(model, 'predict_proba'):
                logger.warning(f"Model '{name}' does not support predict_proba, switching ensemble to 'hard' voting.")
                self.voting = 'hard'
                break

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Fits the ensemble model to the training data."""
        if self.ensemble is None:
            logger.error("Ensemble has not been created or is empty.")
            return {"status": "error", "message": "Ensemble not created."}

        try:
            self.ensemble.fit(X, y)
            self.is_trained = True
            logger.info("Ensemble model trained successfully.")
            return {"status": "success", "message": "Ensemble model trained successfully."}
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}", exc_info=True)
            self.is_trained = False
            return {"status": "error", "message": f"Ensemble training failed: {e}"}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions with the trained ensemble model."""
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction.")
        if self.ensemble is None:
            raise RuntimeError("Ensemble is not available.")
        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities (for classification tasks only)."""
        if self.task_type != "classification":
            raise TypeError("predict_proba is only available for classification tasks.")
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction.")
        if not hasattr(self.ensemble, 'predict_proba'):
             raise AttributeError("The configured ensemble (e.g., with 'hard' voting) does not support predict_proba.")
        return self.ensemble.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluates the model and returns a dictionary of metrics."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        
        preds = self.predict(X)
        
        if self.task_type == "regression":
            self.metrics = {
                "r2_score": r2_score(y, preds),
                "mse": mean_squared_error(y, preds)
            }
        elif self.task_type == "classification":
            self.metrics = {
                "accuracy": accuracy_score(y, preds),
                "f1_score": f1_score(y, preds, average='weighted')
            }
        else:
            self.metrics = {}
            
        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics

    def save_model(self, path: str) -> bool:
        """Saves the ensemble model and its components."""
        if not self.is_trained:
            logger.warning("Attempting to save an untrained ensemble model.")
        
        model_dir = os.path.dirname(path)
        os.makedirs(model_dir, exist_ok=True)
        
        sub_model_configs = []
        for name, model in self.models:
            model_path_prefix = os.path.join(model_dir, f"submodel_{name}")
            if not model.save_model(model_path_prefix):
                logger.error(f"Failed to save sub-model '{name}'. Aborting ensemble save.")
                return False
            
            class_path = f"{model.__class__.__module__}.{model.__class__.__name__}"
            sub_model_configs.append({'name': name, 'class': class_path})

        ensemble_metadata = {
            'task_type': self.task_type,
            'voting': self.voting,
            'sub_models': sub_model_configs,
            'is_trained': self.is_trained
        }
        
        try:
            joblib.dump(ensemble_metadata, path)
            logger.info(f"Ensemble model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ensemble metadata: {e}", exc_info=True)
            return False

    def load_model(self, path: str) -> bool:
        """Loads an ensemble model and its components."""
        try:
            metadata = joblib.load(path)
            self.task_type = metadata['task_type']
            self.voting = metadata.get('voting', 'soft')
            self.is_trained = metadata['is_trained']
            
            model_dir = os.path.dirname(path)
            self.models = []
            
            for model_config in metadata['sub_models']:
                name = model_config['name']
                class_path = model_config['class']
                try:
                    ModelClass = _import_model_class(class_path)
                    model_instance = ModelClass(task_type=self.task_type)
                    
                    model_path_prefix = os.path.join(model_dir, f"submodel_{name}")
                    if not model_instance.load_model(model_path_prefix):
                         logger.warning(f"Failed to load state for sub-model '{name}'.")
                    
                    self.models.append((name, model_instance))
                except Exception as e:
                    logger.error(f"Failed to instantiate or load sub-model '{name}' ({class_path}). Error: {e}")
                    return False

            self.ensemble = self._create_ensemble()
            
            logger.info(f"Ensemble model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ensemble model: {e}", exc_info=True)
            self.is_trained = False
            return False
            
    def get_model_info(self) -> Dict[str, Any]:
        """Returns a dictionary with information about the model."""
        info = {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "is_trained": self.is_trained,
            "voting_strategy": self.voting,
            "sub_models": [model.get_model_info() for _, model in self.models] if self.models else []
        }
        return info
