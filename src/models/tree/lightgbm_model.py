# src/models/tree/lightgbm_model.py

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
import lightgbm as lgb
from src.models.interfaces import BaseModel
from src.core.logging.logger import ProjectLogger

class LightGBMModel(BaseModel):
    """LightGBM model for classification and regression tasks."""

    def __init__(self, n_estimators: int = 100, max_depth: int = -1, learning_rate: float = 0.1, num_leaves: int = 31, task_type: str = "classification", random_state: int = 42):
        super().__init__(model_type="lgbm", task_type=task_type)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.logger = ProjectLogger.get_logger("LightGBMModel")
        self.model = None

    @property
    def name(self) -> str:
        return "lightgbm"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Trains the LightGBM model."""
        try:
            if self.task_type == "classification":
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    random_state=self.random_state,
                    **kwargs
                )
            else:
                self.model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    random_state=self.random_state,
                    **kwargs
                )
            
            self.model.fit(X, y)
            self.is_trained = True
            self.logger.info(f"LightGBM model trained successfully (task: {self.task_type})")
            
            return self.get_model_info()

        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
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
            self.logger.info(f"LightGBM model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Loads a model from a file using joblib."""
        try:
            loaded_model = joblib.load(path)
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"LightGBM model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
