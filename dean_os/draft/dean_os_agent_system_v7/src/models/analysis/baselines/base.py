from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaseBaseline(ABC):
    """Базовий клас для всіх базових (baseline) моделей."""

    def __init__(self, complexity_score: float):
        self.complexity_score = complexity_score

    @abstractmethod
    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        pd.DataFrame | None=None, target_series: pd.Series | None=None
        ) ->dict[str, Any]:
        """Навчає та оцінює базову модель."""
        pass

    def _calculate_metrics(self, actual: pd.Series, predictions: pd.Series
        ) ->dict[str, float]:
        """Розраховує стандартні метрики."""
        return {'mse': float(mean_squared_error(actual, predictions)),
            'mae': float(mean_absolute_error(actual, predictions)), 'r2':
            float(r2_score(actual, predictions))}
