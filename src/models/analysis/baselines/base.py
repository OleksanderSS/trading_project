from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseBaseline(ABC):
    """Базовий клас для всіх базових (baseline) моделей."""

    def __init__(self, complexity_score: float):
        self.complexity_score = complexity_score

    @abstractmethod
    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        Optional[pd.DataFrame]=None, target_series: Optional[pd.Series]=None
        ) ->Dict[str, Any]:
        """Навчає та оцінює базову модель."""
        pass

    def _calculate_metrics(self, actual: pd.Series, predictions: pd.Series
        ) ->Dict[str, float]:
        """Розраховує стандартні метрики."""
        return {'mse': float(mean_squared_error(actual, predictions)),
            'mae': float(mean_absolute_error(actual, predictions)), 'r2':
            float(r2_score(actual, predictions))}
