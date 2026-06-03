from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from .base import BaseBaseline
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger("BaselineModels")


class LinearRegressionBaseline(BaseBaseline):
    """Базова лінійна регресія."""

    def __init__(self, complexity_score: float=1.0, min_samples: int=100):
        super().__init__(complexity_score)
        self.min_samples = min_samples

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        Optional[pd.DataFrame]=None, target_series: Optional[pd.Series]=None
        ) ->Dict[str, Any]:
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
                columns, model.coef_))}
        except Exception as e:
            logger.error(f"Error in LinearRegressionBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"LinearRegressionBaseline training failed: {e}") from e


class SimpleRandomForestBaseline(BaseBaseline):
    """Спрощений випадковий ліс."""

    def __init__(self, complexity_score: float=3.0, min_samples: int=100):
        super().__init__(complexity_score)
        self.min_samples = min_samples

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        Optional[pd.DataFrame]=None, target_series: Optional[pd.Series]=None
        ) ->Dict[str, Any]:
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
                zip(X.columns, model.feature_importances_))}
        except Exception as e:
            logger.error(f"Error in SimpleRandomForestBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"SimpleRandomForestBaseline training failed: {e}") from e
