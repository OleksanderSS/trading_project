# src/models/ensemble/ensemble_model.py

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel


class EnsembleModel(BaseModel):
    """An ensemble model that combines multiple models for improved performance."""

    def __init__(self, models: list[tuple[str, Any]], task_type: str = "classification", voting: str = "soft"):
        super().__init__(model_type="ensemble", task_type=task_type)
        self.models = models
        self.voting = voting
        self.logger = ProjectLogger.get_logger("EnsembleModel")
        self.ensemble: VotingClassifier | VotingRegressor | None = None

    @property
    def name(self) -> str:
        return "ensemble"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict[str, Any]:
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
        if not self.is_trained or self.ensemble is None:
            raise ValueError("Model must be trained before prediction.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities (for classification tasks)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        if not self.is_trained or self.ensemble is None:
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
                self.logger.warning(f"🚫 Blocking unsafe ensemble load attempt from: {path}")
                raise ValueError(f"Unsafe path for loading: {path}")

            loaded_model = joblib.load(trusted_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"Ensemble model loaded from {trusted_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
