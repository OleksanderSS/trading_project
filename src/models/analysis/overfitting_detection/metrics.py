from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class OverfittingMetrics:
    """Calculates metrics for overfitting detection."""

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate various regression metrics."""
        try:
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating metrics (returning empty): {e}")
            return {}

    def analyze_data_characteristics(self,
                                 X_train: pd.DataFrame,
                                 X_val: pd.DataFrame | None) -> dict[str, Any]:
        """Analyze data dimensions and characteristics."""
        chars = {
            'n_train_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns)
        }
        if X_val is not None:
            chars['n_val_samples'] = X_val.shape[0]
            chars['val_ratio'] = X_val.shape[0] / X_train.shape[0]
        return chars
