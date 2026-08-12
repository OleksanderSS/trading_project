# src/models/linear/linear_model.py

from typing import Any

import joblib
import numpy as np
import pandas as pd
import numpy as _np
from sklearn.linear_model import LogisticRegression, RidgeCV

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel


class LinearModel(BaseModel):
    """Linear model for regression and classification tasks."""

    def __init__(self, task_type: str = "regression"):
        super().__init__(model_type="linear", task_type=task_type)
        self.logger = ProjectLogger.get_logger("LinearModel")
        self.model = None

    @property
    def name(self) -> str:
        return "linear"

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict[str, Any]:
        """Trains the linear model."""
        try:
            if self.task_type == "classification":
                self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, **kwargs)
            else:
                # Was LinearRegression -- unregularised OLS, on 35 correlated
                # features and ~306 training rows. Measured on the 2026-08-12
                # batch across 22 daily contexts, median holdout R2:
                #
                #     OLS,        35 features   -7.35
                #     RidgeCV,    35 features   -0.85
                #     RidgeCV,    10 features   -0.25
                #     ElasticNet, 35 features   -0.14
                #     baseline (predict the training mean)  -0.01
                #
                # A model at -7.35 is not weak, it is worse than a constant by
                # a factor of hundreds, and `linear` had won 135 of 354
                # champion slots. Note the asymmetry this fixes: the
                # classification branch above has always been regularised
                # (sklearn's LogisticRegression applies L2 by default), so
                # only regression targets were exposed.
                #
                # RidgeCV picks alpha by leave-one-out over the training rows,
                # so there is no hyperparameter to guess and no validation
                # data spent. It does NOT make returns predictable -- even
                # ElasticNet stays below the baseline on the median -- it
                # stops this model from actively fabricating fits.
                kwargs.setdefault('alphas', _np.logspace(-3, 4, 40))
                self.model = RidgeCV(**kwargs)

            self.model.fit(X, y)
            self.is_trained = True
            self.logger.info(f"Linear model trained successfully (task: {self.task_type})")

            return self.get_model_info()

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
                self.logger.warning(f"🚫 Blocking unsafe Linear model load attempt from: {path}")
                raise ValueError(f"Unsafe path for loading: {path}")

            loaded_model = joblib.load(trusted_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            self.__dict__.update(loaded_model.__dict__)
            self.logger.info(f"Linear model loaded from {trusted_path}")
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
