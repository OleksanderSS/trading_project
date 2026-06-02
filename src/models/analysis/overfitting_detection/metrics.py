import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class OverfittingMetrics:
    """Calculates metrics for overfitting detection."""
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various regression metrics."""
        try:
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            raise RuntimeError("Failed to calculate overfitting metrics") from e

    def analyze_data_characteristics(self, 
                                 X_train: pd.DataFrame, 
                                 X_val: Optional[pd.DataFrame]) -> Dict[str, Any]:
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
