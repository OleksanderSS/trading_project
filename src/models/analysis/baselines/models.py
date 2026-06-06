from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

from .base import BaseBaseline

logger = ProjectLogger.get_logger("BaselineModels")


class LinearRegressionBaseline(BaseBaseline):
    """Базова лінійна регресія."""

    def __init__(self, complexity_score: float=1.0, min_samples: int=100):
        super().__init__(complexity_score)
        self.min_samples = min_samples

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        pd.DataFrame | None=None, target_series: pd.Series | None=None
        ) ->dict[str, Any]:
        try:
            if features_df is None or target_series is None or len(features_df
                ) < self.min_samples:
                return {'model_type': 'linear_regression', 'status':
                    'insufficient_data'}
            X = features_df.select_dtypes(include=[np.number]).fillna(
                features_df.mean())
            if len(X.columns) == 0:
                return {'model_type': 'linear_regression', 'status':
                    'no_numeric_features'}
            model = LinearRegression()
            model.fit(X, target_series)
            predictions = pd.Series(model.predict(X), index=target_series.index)
            return {'model_type': 'linear_regression', 'predictions':
                predictions, 'metrics': self._calculate_metrics(target_series,
                predictions), 'complexity_score': self.complexity_score,
                'feature_count': len(X.columns), 'coefficients': dict(zip(X.
                columns, model.coef_, strict=False))}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error in LinearRegressionBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"LinearRegressionBaseline training failed: {e}") from e


class SimpleRandomForestBaseline(BaseBaseline):
    """Спрощений випадковий ліс."""

    def __init__(self, complexity_score: float=3.0, min_samples: int=100):
        super().__init__(complexity_score)
        self.min_samples = min_samples

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        pd.DataFrame | None=None, target_series: pd.Series | None=None
        ) ->dict[str, Any]:
        try:
            if features_df is None or target_series is None or len(features_df
                ) < self.min_samples:
                return {'model_type': 'random_forest_simple', 'status':
                    'insufficient_data'}
            X = features_df.select_dtypes(include=[np.number]).fillna(
                features_df.mean())
            if len(X.columns) == 0:
                return {'model_type': 'random_forest_simple', 'status':
                    'no_numeric_features'}
            model = RandomForestRegressor(n_estimators=10, max_depth=5,
                random_state=42, n_jobs=-1)
            model.fit(X, target_series)
            predictions = pd.Series(model.predict(X), index=target_series.index)
            return {'model_type': 'random_forest_simple', 'predictions':
                predictions, 'metrics': self._calculate_metrics(target_series,
                predictions), 'complexity_score': self.complexity_score,
                'feature_count': len(X.columns), 'feature_importance': dict(
                zip(X.columns, model.feature_importances_, strict=False))}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error in SimpleRandomForestBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"SimpleRandomForestBaseline training failed: {e}") from e
