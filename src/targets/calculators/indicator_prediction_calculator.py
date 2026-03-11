
import pandas as pd
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("IndicatorPredictionCalculator")

class IndicatorPredictionCalculator:
    """
    Calculates targets by shifting existing indicator columns.
    """
    def calculate(self, df: pd.DataFrame, indicator_col: str, shift: int, **kwargs) -> pd.Series:
        """
        Shifts the specified indicator column to create a future target.

        Args:
            df (pd.DataFrame): The input DataFrame.
            indicator_col (str): The column of the indicator to be shifted.
            shift (int): The number of periods to look into the future (should be negative).

        Returns:
            pd.Series: The shifted indicator series.
        """
        if indicator_col not in df.columns:
            logger.warning(f"Indicator column '{indicator_col}' not found for target generation. Returning NaNs.")
            return pd.Series(index=df.index, dtype=float)
            
        target_series = df[indicator_col].shift(shift)
        return target_series
