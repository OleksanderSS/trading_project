import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from .base import BaseEnricher

logger = logging.getLogger(__name__)

class DecayFeaturesEnricher(BaseEnricher):
    """
    Enricher that calculates exponentially decaying influence features from discrete event flags.
    Based on the concept that the impact of a market event (e.g., a news shock or price anomaly) 
    is highest at the moment of occurrence and fades over time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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

    def _enrich_impl(self, df: pd.DataFrame, event_columns: Optional[List[str]] = None, half_life_periods: Optional[int] = None, **kwargs) -> pd.DataFrame:
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

        # Calculate decay factor based on half-life formula: N(t) = N0 * e^(-λt)
        # Multiplier per step = exp(-ln(2) / half_life)
        decay_factor = np.exp(-np.log(2) / half_life_periods)
        
        enriched_df = df.copy()

        for col in event_columns:
            if col not in df.columns:
                logger.warning(f"Event column '{col}' not found in DataFrame. Skipping.")
                continue

            # Vectorized approach for exponential decay with resets
            # 1. Identify indices where events occur
            is_event = df[col].values >= 1
            event_indices = np.where(is_event)[0]
            
            decayed_values = np.zeros(len(df))
            
            # 2. Apply decay.
            # This is equivalent to finding the last event index for each position
            # and applying decay based on the distance to that event.
            if len(event_indices) > 0:
                # Fill positions with 0 initially, they will be updated if events exist
                # Create an array of last event indices
                last_event_indices = np.zeros(len(df), dtype=int)
                
                # Fill last_event_indices efficiently
                idx = 0
                for i in range(len(df)):
                    if idx < len(event_indices) and event_indices[idx] <= i:
                        last_event_idx = event_indices[idx]
                        if idx + 1 < len(event_indices) and event_indices[idx+1] <= i:
                            idx += 1
                            continue # Wait for next event
                    else:
                         # This logic needs to be careful
                         pass
                
                # Actually, the simplest efficient vectorized way without complex numpy:
                # Just use a Numba-jitted function or keep the loop if performance is acceptable.
                # Given the constraints, let's keep the loop but optimize it slightly.
                # The current loop is already quite simple.
                
                # Let's optimize the loop by using numpy directly in it.
                decayed_values = np.zeros(len(df))
                current_value = 0.0
                values = df[col].values
                for i in range(len(df)):
                    if values[i] >= 1:
                        current_value = 1.0
                    else:
                        current_value *= decay_factor
                    decayed_values[i] = current_value

            enriched_df[f"{col}_decayed"] = decayed_values
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Added decay feature for '{col}' with half-life {half_life_periods}.")

        return enriched_df