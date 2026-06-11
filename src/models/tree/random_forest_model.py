# src/models/tree/random_forest_model.py

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel


class RandomForestModel(BaseModel):
    """RandomForest model for classification and regression tasks."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, task_type: str = "classification", random_state: int = 42):
        super().__init__(model_type="rf", task_type=task_type)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.logger = ProjectLogger.get_logger("RandomForestModel")
        self.model = None

    @property
    def name(self) -> str:
        return "random_forest"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict[str, Any]:
        """Trains the RandomForest model."""
        try:
            if isinstance(X, pd.DataFrame):
                self.feature_cols = X.columns.tolist()

            if self.task_type == "classification":
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    **kwargs
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    **kwargs
                )

            self.model.fit(X, y)
            self.is_trained = True
            self.logger.info(f"RandomForest model trained successfully (task: {self.task_type})")

            return self.get_model_info()

        except Exception as e:
            self.logger.error(f"RandomForest training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

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
            self.logger.info(f"RandomForest model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Loads a model from a file using joblib."""
        try:
            loaded_model = joblib.load(path)
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"RandomForest model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
