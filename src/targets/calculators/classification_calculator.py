

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ClassificationCalculator")

class ClassificationCalculator:
    """
    Calculates binary and multiclass classification targets.
    """
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Unified entry point for classification target calculation.
        """
        # Create a copy to avoid mutating the original config
        params = kwargs.copy()

        # Extract common params
        base_col = params.pop('base_col', 'close')
        shift = params.pop('shift', -1)

        if 'thresholds' in params:
            thresholds = params.pop('thresholds')
            return self.calculate_multiclass(df, base_col, shift, thresholds, **params)
        else:
            # Default to binary with 0.0 threshold if not specified
            threshold = params.pop('threshold', 0.0)
            return self.calculate_binary(df, base_col, shift, threshold, **params)

    def calculate_binary(self, df: pd.DataFrame, base_col: str, shift: int, threshold: float, **kwargs) -> pd.Series:
        """
        Generates a binary target: 1 if future return > threshold, else 0.
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        future_price = df[base_col].shift(shift)
        returns = (future_price - df[base_col]) / df[base_col]

        target_series = (returns > threshold).astype(int)
        target_series[returns.isna()] = np.nan # Propagate NaNs
        return target_series

    def calculate_multiclass(self, df: pd.DataFrame, base_col: str, shift: int, thresholds: list[float], **kwargs) -> pd.Series:
        """
        Generates a multiclass target based on return thresholds.
        e.g., [-0.01, 0.01] -> 0 (Down), 1 (Flat), 2 (Up)
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        future_price = df[base_col].shift(shift)
        returns = (future_price - df[base_col]) / df[base_col]

        # ✅ Upgraded to use np.digitize to support arbitrary length of thresholds (N-class binning)
        # digitize handles any number of bins robustly without hardcoding index offsets
        bins = sorted(thresholds)
        digitized = np.digitize(returns.values, bins)
        
        target_series = pd.Series(digitized, index=df.index, dtype=float)
        target_series[returns.isna()] = np.nan # Propagate NaNs
        return target_series
