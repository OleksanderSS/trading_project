# src/models/interfaces.py - Unified interface for all models

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.metrics.calculator import MetricsCalculator
from src.utils.artifact_security import resolve_trusted_artifact_path


class BaseModel(ABC):
    """Abstract base class for all models, defining a unified interface."""

    def __init__(self, model_type: str, task_type: str = "regression"):
        self.model_type = model_type
        self.task_type = task_type
        self.is_trained = False
        self.feature_cols = None
        self.metrics = {}
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    @property
    def name(self) -> str:
        """Returns unique model name."""
        return f"{self.model_type}_{self.task_type}"

    @abstractmethod
    def train(
        self, X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs
    ) -> dict[str, Any]:
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Makes predictions."""
        pass

    def evaluate(
        self, X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        task_type: str | None = None
    ) -> dict[str, float]:
        """
        Evaluates model performance using centralized metrics calculator.
        
        Args:
            X: Feature data
            y: True labels
            task_type: Task type ('classification' or 'regression'). If None, uses self.task_type.
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Evaluating model {self.name}...")
        predictions = self.predict(X)

        # Use provided task_type or fall back to instance task_type
        effective_task_type = task_type if task_type is not None else self.task_type

        calculator = MetricsCalculator()

        # For classification, try to get probability predictions if available
        y_prob = None
        if effective_task_type == 'classification' and hasattr(self, 'predict_proba'):
            try:
                y_prob = self.predict_proba(X)
            except Exception:
                # If predict_proba fails, continue without probabilities
                pass

        # Use unified calculator for ML metrics with task_type instead of is_classification
        results = calculator.get_ml_metrics(
            y, predictions, y_prob=y_prob, task_type=effective_task_type
        )

        self.metrics.update(results)
        return results

    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Saves model to file."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Loads model from file."""
        pass

    def _resolve_model_artifact_path(
        self,
        path: str | Path,
        *,
        allowed_suffixes: set[str] | None = None,
        must_exist: bool = True,
    ) -> Path:
        """Resolve a trusted model artifact path before deserialization."""
        return resolve_trusted_artifact_path(
            path,
            allowed_suffixes=allowed_suffixes,
            must_exist=must_exist,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Returns model information."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "is_trained": self.is_trained,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics
        }
