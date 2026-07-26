import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseEnricher

logger = logging.getLogger(__name__)

class DecayFeaturesEnricher(BaseEnricher):
    """
    Enricher that calculates exponentially decaying influence features from discrete event flags.
    Based on the concept that the impact of a market event (e.g., a news shock or price anomaly)
    is highest at the moment of occurrence and fades over time.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with optional config dict from FeatureOrchestrator"""
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        self.config = config or {}
        self.half_life_periods = self.config.get('half_life_periods', 20)
        self.default_event_columns = self.config.get('event_columns', ['is_significant'])
        logger.info(f"DecayFeaturesEnricher initialized with half_life={self.half_life_periods} periods")

    @property
    def name(self) -> str:
        return "decay_features"

    @property
    def priority(self) -> int:
        """Execution order - run after significance features (70)"""
        return 75

    def _calculate_decay_factor(self, half_life_periods: int) -> float:
        """Calculate decay factor based on half-life formula."""
        return np.exp(-np.log(2) / half_life_periods)

    def _apply_decay_to_column(self, df: pd.DataFrame, col: str, decay_factor: float) -> np.ndarray:
        """Apply exponential decay to a single column.

        The decay state is sequential/stateful, so it must reset at ticker
        boundaries - otherwise a recent event in one ticker's last rows
        leaks a nonzero decayed value into the next ticker's first rows.

        Returns a plain positional numpy array (length == len(df)), not an
        index-aligned Series: multiple tickers naturally share the same
        trading dates, so df's index is commonly non-unique by this point
        in the pipeline (e.g. a shared DatetimeIndex) - reindex()/index
        based reassembly breaks or silently misaligns against duplicate
        labels, while boolean-mask numpy assignment is purely positional
        and unaffected by the index at all.
        """
        if col not in df.columns:
            logger.warning(f"Event column '{col}' not found in DataFrame. Skipping.")
            return np.zeros(len(df))

        values = df[col].to_numpy()
        decayed_values = np.zeros(len(df))

        if 'ticker' in df.columns:
            tickers = df['ticker'].to_numpy()
            for ticker in pd.unique(tickers):
                mask = tickers == ticker
                decayed_values[mask] = self._decay_single_array(values[mask], decay_factor)
        else:
            decayed_values = self._decay_single_array(values, decay_factor)

        return decayed_values

    def _decay_single_array(self, values: np.ndarray, decay_factor: float) -> np.ndarray:
        """Apply exponential decay to a single (already single-ticker) array."""
        decayed_values = np.zeros(len(values))
        current_value = 0.0
        for i in range(len(values)):
            if values[i] >= 1:
                current_value = 1.0
            else:
                current_value *= decay_factor
            decayed_values[i] = current_value
        return decayed_values

    def _enrich_impl(self, df: pd.DataFrame, event_columns: list[str] | None = None, half_life_periods: int | None = None, **kwargs) -> pd.DataFrame:
        """
        Adds exponential decay features for specified event columns.

        Args:
            df (pd.DataFrame): The input DataFrame containing event flags (1 for event, 0 otherwise).
            event_columns (list): List of column names to apply decay to. If None, uses config defaults.
            half_life_periods (int): The number of periods it takes for the signal to reach 0.5. If None, uses config defaults.
            **kwargs: Additional parameters.

        Returns:
            pd.DataFrame: DataFrame with added '{column}_decayed' columns.
        """
        if df.empty:
            logger.warning("DecayFeaturesEnricher received an empty DataFrame.")
            return df

        # Use config defaults if parameters not provided
        if event_columns is None:
            event_columns = self.default_event_columns
            logger.info(f"Using default event_columns from config: {event_columns}")

        if half_life_periods is None:
            half_life_periods = self.half_life_periods
            logger.info(f"Using default half_life_periods from config: {half_life_periods}")

        decay_factor = self._calculate_decay_factor(half_life_periods)

        enriched_df = df.copy()

        for col in event_columns:
            decayed_values = self._apply_decay_to_column(df, col, decay_factor)
            enriched_df[f"{col}_decayed"] = decayed_values
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Added decay feature for '{col}' with half-life {half_life_periods}.")

        return enriched_df
