
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("IndicatorPredictionCalculator")

class IndicatorPredictionCalculator:
    """
    Calculates targets by shifting existing indicator columns.
    """

    #: Params this calculator honours — see RegressionCalculator for why this
    #: is declared rather than introspected.
    SUPPORTED_PARAMS = frozenset({"indicator_col", "shift", "source_timeframe"})
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

        if shift >= 0:
            logger.error(f"Shift must be negative for future targets. Got shift={shift}.")
            raise ValueError(f"Shift must be negative for future targets. Got shift={shift}.")

        # Shift per-ticker so a multi-ticker frame never leaks the next
        # ticker's indicator value into the previous ticker's future target.
        if "ticker" in df.columns:
            target_series = df.groupby("ticker")[indicator_col].shift(shift)
        else:
            target_series = df[indicator_col].shift(shift)
        return target_series
