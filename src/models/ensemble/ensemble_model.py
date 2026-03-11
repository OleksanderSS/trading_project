# src/models/ensemble/ensemble_model.py

import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple, Any, Dict
from sklearn.ensemble import VotingClassifier, VotingRegressor
from src.models.interfaces import BaseModel
from src.core.logging.logger import ProjectLogger

class EnsembleModel(BaseModel):
    """An ensemble model that combines multiple models for improved performance."""

    def __init__(self, models: List[Tuple[str, Any]], task_type: str = "classification", voting: str = "soft"):
        super().__init__(model_type="ensemble", task_type=task_type)
        self.models = models
        self.voting = voting
        self.logger = ProjectLogger.get_logger("EnsembleModel")
        self.ensemble = None

    @property
    def name(self) -> str:
        return "ensemble"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Trains the ensemble model."""
        try:
            if self.task_type == "classification":
                self.ensemble = VotingClassifier(
                    estimators=self.models,
                    voting=self.voting
                )
            else:
                self.ensemble = VotingRegressor(estimators=self.models)
            
            self.ensemble.fit(X, y)
            self.is_trained = True
            self.logger.info(f"Ensemble model trained successfully with {len(self.models)} models.")
            
            return self.get_model_info()

        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions with the trained ensemble model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        return self.ensemble.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities (for classification tasks)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.ensemble.predict_proba(X)

    def save_model(self, path: str) -> bool:
        """Saves the ensemble model to a file."""
        if not self.is_trained:
            self.logger.error("Cannot save an untrained model.")
            return False
        
        try:
            joblib.dump(self, path)
            self.logger.info(f"Ensemble model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Loads an ensemble model from a file."""
        try:
            loaded_model = joblib.load(path)
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"Ensemble model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
