# src/models/linear/linear_model.py

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, LogisticRegression
from src.models.interfaces import BaseModel
from src.core.logging.logger import ProjectLogger

class LinearModel(BaseModel):
    """Linear model for regression and classification tasks."""

    def __init__(self, task_type: str = "regression"):
        super().__init__(model_type="linear", task_type=task_type)
        self.logger = ProjectLogger.get_logger("LinearModel")
        self.model = None

    @property
    def name(self) -> str:
        return "linear"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Trains the linear model."""
        try:
            if self.task_type == "classification":
                self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, **kwargs)
            else:
                self.model = LinearRegression(**kwargs)
            
            self.model.fit(X, y)
            self.is_trained = True
            self.logger.info(f"Linear model trained successfully (task: {self.task_type})")
            
            return self.get_model_info()

        except Exception as e:
            self.logger.error(f"Linear model training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities (for classification tasks)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)

    def save_model(self, path: str) -> bool:
        """Saves the model to a file using joblib."""
        if not self.is_trained:
            self.logger.error("Cannot save an untrained model.")
            return False
        
        try:
            joblib.dump(self, path)
            self.logger.info(f"Linear model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Loads a model from a file using joblib."""
        try:
            loaded_model = joblib.load(path)
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"Linear model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
