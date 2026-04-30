"""
Розрахунок метрик для оцінки моделей
"""
from typing import Dict, Tuple
import numpy as np


class MetricsCalculator:
    """Розрахунок метрик для оцінки моделей"""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Розрахувати метрики якості"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }

    @staticmethod
    def compute_data_signature(df_feat: object, df_targ: object) -> str:
        """Розрахувати сигнатуру даних для кешування"""
        import hashlib
        import pandas as pd
        
        # Use SHA-256 instead of MD5 for better security
        feat_hash = hashlib.sha256(pd.util.hash_pandas_object(df_feat, index=True).values).hexdigest()
        targ_hash = hashlib.sha256(pd.util.hash_pandas_object(df_targ, index=True).values).hexdigest()
        
        return f"{feat_hash}_{targ_hash}"

    @staticmethod
    def calculate_batch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Розрахувати метрики для батчу"""
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'batch_mse': float(mse),
            'batch_mae': float(mae)
        }
