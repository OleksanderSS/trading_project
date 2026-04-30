# src/models/interfaces.py - Unified interface for all models

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from src.metrics.calculator import MetricsCalculator
from src.core.logging.logger import ProjectLogger


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
        self, X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """Trains the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Makes predictions."""
        pass
    
    def evaluate(
        self, X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """Evaluates model performance using centralized metrics calculator."""
        self.logger.info(f"Evaluating model {self.name}...")
        predictions = self.predict(X)
        
        calculator = MetricsCalculator()
        is_classification = (self.task_type == 'classification')
        
        # Use unified calculator for ML metrics
        results = calculator.get_ml_metrics(
            y, predictions, is_classification=is_classification
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returns model information."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "is_trained": self.is_trained,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics
        }