

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.targets.calculators.future_shift import future_shift

logger = ProjectLogger.get_logger("ClassificationCalculator")

class ClassificationCalculator:
    """
    Calculates binary and multiclass classification targets.
    """
    def calculate(self, df: pd.DataFrame, base_col: str, shift: int, **kwargs) -> pd.Series:
        """
        Interface entry point shared with the other target calculators.

        ``TargetOrchestrator`` dispatches to :meth:`calculate_binary` /
        :meth:`calculate_multiclass` by name, but ``TemporalTargetGuard`` and the
        contract tests call ``calculate`` like they do on every other calculator.
        Without this method that call raised ``AttributeError``, which the guard
        caught and turned into a silently missing target.

        Dispatches to multiclass when ``thresholds`` is supplied, binary otherwise.
        """
        thresholds = kwargs.pop('thresholds', None)
        if thresholds is not None:
            return self.calculate_multiclass(df, base_col, shift, thresholds, **kwargs)
        threshold = kwargs.pop('threshold', 0.0)
        return self.calculate_binary(df, base_col, shift, threshold, **kwargs)

    def calculate_binary(self, df: pd.DataFrame, base_col: str, shift: int, threshold: float, **kwargs) -> pd.Series:
        """
        Generates a binary target: 1 if future return > threshold, else 0.
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        if shift >= 0:
            logger.error(f"Shift must be negative for future targets. Got shift={shift}.")
            raise ValueError(f"Shift must be negative for future targets. Got shift={shift}.")

        future_price = future_shift(df, base_col, shift)
        returns = (future_price - df[base_col]) / df[base_col]

        target_series = pd.Series(
            np.where(returns.isna(), np.nan, (returns > threshold).astype(float)),
            index=df.index
        )
        return target_series

    def calculate_multiclass(self, df: pd.DataFrame, base_col: str, shift: int, thresholds: list[float], **kwargs) -> pd.Series:
        """
        Generates a multiclass target based on return thresholds.
        e.g., [-0.01, 0.01] -> 0 (Down), 1 (Flat), 2 (Up)
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        if shift >= 0:
            logger.error(f"Shift must be negative for future targets. Got shift={shift}.")
            raise ValueError(f"Shift must be negative for future targets. Got shift={shift}.")

        future_price = future_shift(df, base_col, shift)
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
