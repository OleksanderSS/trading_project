
import pandas as pd
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RegressionCalculator")

class RegressionCalculator:
    """
    Calculates regression targets based on future returns.
    """
    def calculate(self, df: pd.DataFrame, base_col: str, shift: int, **kwargs) -> pd.Series:
        """
        Calculates the future percentage return.

        Args:
            df (pd.DataFrame): The input DataFrame.
            base_col (str): The column to use for calculation (e.g., 'close').
            shift (int): The number of periods to look into the future (should be negative).

        Returns:
            pd.Series: A series with the calculated future returns.
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found in DataFrame.")
            raise ValueError(f"Base column '{base_col}' not found.")
            
        future_price = df[base_col].shift(shift)
        target_series = (future_price - df[base_col]) / df[base_col]
        
        return target_series
