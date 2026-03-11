# src/models/interfaces.py - Уніфікований інтерфейс для всіх моделей

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from src.metrics.calculator import MetricsCalculator
from src.core.logging.logger import ProjectLogger

class BaseModel(ABC):
    """Абстрактний базовий клас для всіх моделей, що визначає уніфікований інтерфейс."""
    
    def __init__(self, model_type: str, task_type: str = "regression"):
        self.model_type = model_type
        self.task_type = task_type
        self.is_trained = False
        self.feature_cols = None
        self.metrics = {}
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        """Повертає унікальне ім'я моделі."""
        return f"{self.model_type}_{self.task_type}"

    @abstractmethod
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """Тренує модель."""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Робить прогнози."""
        pass
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Оцінює продуктивність моделі за допомогою централізованого калькулятора метрик."""
        self.logger.info(f"Оцінка моделі {self.name}...")
        predictions = self.predict(X)
        
        calculator = MetricsCalculator()
        is_classification = (self.task_type == 'classification')
        
        # Використовуємо уніфікований калькулятор для отримання ML метрик
        results = calculator.get_ml_metrics(y, predictions, is_classification=is_classification)
        
        self.metrics.update(results)
        return results
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Зберігає модель у файл."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Завантажує модель з файлу."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Повертає інформацію про модель."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "task_type": self.task_type,
            "is_trained": self.is_trained,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics
        }