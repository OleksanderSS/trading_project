# src/models/tree/lightgbm_model.py

from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel


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

        # ✅ INITIALIZE MODEL IN __INIT__ (for Ensemble)
        if self.task_type == "classification":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state
            )

    @property
    def name(self) -> str:
        return "lightgbm"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict[str, Any]:
        """Trains the LightGBM model."""
        try:
            eval_set = kwargs.pop("eval_set", None)
            early_stopping_rounds = kwargs.pop("early_stopping_rounds", None)

            # Update parameters if they are provided
            if kwargs:
                self.model.set_params(**kwargs)

            fit_kwargs = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = eval_set
            
            if early_stopping_rounds is not None:
                fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds)]

            self.model.fit(X, y, **fit_kwargs)
            self.is_trained = True
            self.logger.info(f"LightGBM model trained successfully (task: {self.task_type})")

            return self.get_model_info()

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
                self.logger.warning(f"🚫 Blocking unsafe LightGBM model load attempt from: {path}")
                raise ValueError(f"Unsafe path for loading: {path}")

            loaded_model = joblib.load(trusted_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"LightGBM model loaded from {trusted_path}")
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
