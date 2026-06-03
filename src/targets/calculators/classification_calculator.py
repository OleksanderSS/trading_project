
import pandas as pd
import numpy as np
from typing import List
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ClassificationCalculator")

class ClassificationCalculator:
    """
    Calculates binary and multiclass classification targets.
    """
    def calculate_binary(self, df: pd.DataFrame, base_col: str, shift: int, threshold: float, **kwargs) -> pd.Series:
        """
        Generates a binary target: 1 if future return > threshold, else 0.
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        future_price = df[base_col].shift(shift)
        returns = (future_price - df[base_col]) / df[base_col]
        
        target_series = pd.Series(
            np.where(returns.isna(), np.nan, (returns > threshold).astype(float)),
            index=df.index
        )
        return target_series

    def calculate_multiclass(self, df: pd.DataFrame, base_col: str, shift: int, thresholds: List[float], **kwargs) -> pd.Series:
        """
        Generates a multiclass target based on return thresholds.
        e.g., [-0.01, 0.01] -> 0 (Down), 1 (Flat), 2 (Up)
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        future_price = df[base_col].shift(shift)
        returns = (future_price - df[base_col]) / df[base_col]

        # Use np.select for clear, vectorized logic
        conditions = [
            returns <= thresholds[0],
            (returns > thresholds[0]) & (returns < thresholds[1]),
            returns >= thresholds[1]
        ]
        choices = [0, 1, 2] # Down, Flat, Up
        
        target_series = pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)
        target_series[returns.isna()] = np.nan # Propagate NaNs
        return target_series
