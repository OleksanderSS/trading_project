

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ClassificationCalculator")

class ClassificationCalculator:
    """
    Calculates binary and multiclass classification targets.
    """

    #: Params this calculator honours — see RegressionCalculator for why this
    #: is declared rather than introspected.
    SUPPORTED_PARAMS = frozenset({
        "base_col", "shift", "threshold", "thresholds",
        "compare_to", "window", "indicator_col",
    })
    def calculate_binary(self, df: pd.DataFrame, base_col: str, shift: int, threshold: float, **kwargs) -> pd.Series:
        """Binary target. Three comparison modes, selected by params:

        - default: 1 if the future RELATIVE CHANGE of `base_col` exceeds
          `threshold`. Note `threshold` is a fraction, not a multiple:
          0.005 means +0.5%, and 2.0 would mean +200%.
        - `compare_to: "average"`: 1 if the future value of `base_col` exceeds
          `threshold` TIMES its trailing rolling mean (window from `window`,
          default 20). Here `threshold` IS a multiple, so 2.0 means "twice the
          average" -- which is what a "volume spike > 2x average" target
          actually means. Previously this param was accepted and silently
          ignored, so the comparison ran against the current bar and a
          threshold of 2.0 demanded +200% (a tripling) instead.
        - `indicator_col`: 1 if the future value of `base_col` exceeds that
          indicator's CURRENT value by more than `threshold` (a fraction).
          This is the Bollinger-style breakout case: "does price close above
          today's upper band within the horizon". Previously `indicator_col`
          was ignored here too, leaving a plain direction target.
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found.")
            raise ValueError(f"Base column '{base_col}' not found.")

        if shift >= 0:
            logger.error(f"Shift must be negative for future targets. Got shift={shift}.")
            raise ValueError(f"Shift must be negative for future targets. Got shift={shift}.")

        # Shift per-ticker so a multi-ticker frame never leaks the next
        # ticker's price into the previous ticker's future-return target.
        if "ticker" in df.columns:
            future_price = df.groupby("ticker")[base_col].shift(shift)
        else:
            future_price = df[base_col].shift(shift)

        compare_to = kwargs.get("compare_to")
        indicator_col = kwargs.get("indicator_col")

        if compare_to == "average":
            window = int(kwargs.get("window", 20))
            if "ticker" in df.columns:
                reference = df.groupby("ticker")[base_col].transform(
                    lambda s: s.rolling(window, min_periods=window).mean()
                )
            else:
                reference = df[base_col].rolling(window, min_periods=window).mean()
            ratio = future_price / reference.replace(0, np.nan)
            hit = ratio > threshold
            invalid = ratio.isna()
        elif indicator_col:
            if indicator_col not in df.columns:
                logger.error(
                    f"indicator_col '{indicator_col}' not found; cannot build "
                    f"an indicator-crossing target."
                )
                raise ValueError(f"Indicator column '{indicator_col}' not found.")
            reference = df[indicator_col]
            excess = (future_price - reference) / reference.replace(0, np.nan)
            hit = excess > threshold
            invalid = excess.isna()
        else:
            returns = (future_price - df[base_col]) / df[base_col]
            hit = returns > threshold
            invalid = returns.isna()

        target_series = pd.Series(
            np.where(invalid, np.nan, hit.astype(float)),
            index=df.index,
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

        # Shift per-ticker so a multi-ticker frame never leaks the next
        # ticker's price into the previous ticker's future-return target.
        if "ticker" in df.columns:
            future_price = df.groupby("ticker")[base_col].shift(shift)
        else:
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
