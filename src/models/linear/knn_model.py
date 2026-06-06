# src/models/linear/knn_model.py

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel


class KNNModel(BaseModel):
    """K-Nearest Neighbors model for classification and regression tasks."""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", task_type: str = "classification"):
        super().__init__(model_type="knn", task_type=task_type)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.logger = ProjectLogger.get_logger("KNNModel")
        self.model = None

    @property
    def name(self) -> str:
        return "knn"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict[str, Any]:
        """Trains the KNN model."""
        try:
            if self.task_type == "classification":
                self.model = KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    **kwargs
                )
            else:
                self.model = KNeighborsRegressor(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    **kwargs
                )

            self.model.fit(X, y)
            self.is_trained = True
            self.logger.info(f"KNN model trained successfully (task: {self.task_type}, n_neighbors={self.n_neighbors})")

            return self.get_model_info()

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"KNN training failed: {e}")
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
            self.logger.info(f"KNN model saved to {path}")
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Loads a model from a file using joblib."""
        try:
            from pathlib import Path

            from src.config.unified_config_manager import get_current_config
            from src.utils.artifact_security import resolve_trusted_artifact_path

            # Security validation: Ensure path is within expected data or models directories
            trusted_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={'.joblib', '.pkl', '.pickle'},
                must_exist=True,
            )

            # Validate against configured model storage paths
            config = get_current_config()
            base_model_path = config.get('models.dual_model_manager.base_path', 'data/models')

            if not trusted_path.resolve().is_relative_to(Path(base_model_path).resolve()):
                self.logger.warning(f"🚫 Blocking unsafe KNN model load attempt from: {path}")
                raise ValueError(f"Unsafe path for loading: {path}")

            loaded_model = joblib.load(trusted_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"KNN model loaded from {trusted_path}")
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
