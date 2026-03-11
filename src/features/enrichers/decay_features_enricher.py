import numpy as np
import pandas as pd
import logging
from .base import BaseEnricher

logger = logging.getLogger(__name__)

class DecayFeaturesEnricher(BaseEnricher):
    """
    Enricher that calculates exponentially decaying influence features from discrete event flags.
    Based on the concept that the impact of a market event (e.g., a news shock or price anomaly) 
    is highest at the moment of occurrence and fades over time.
    """

    def enrich(self, df: pd.DataFrame, event_columns: list, half_life_periods: int = 20, **kwargs) -> pd.DataFrame:
        """
        Adds exponential decay features for specified event columns.

        Args:
            df (pd.DataFrame): The input DataFrame containing event flags (1 for event, 0 otherwise).
            event_columns (list): List of column names to apply decay to.
            half_life_periods (int): The number of periods it takes for the signal to reach 0.5.
            **kwargs: Additional parameters.

        Returns:
            pd.DataFrame: DataFrame with added '{column}_decayed' columns.
        """
        if df.empty:
            logger.warning("DecayFeaturesEnricher received an empty DataFrame.")
            return df

        # Calculate decay factor based on half-life formula: N(t) = N0 * e^(-λt)
        # Multiplier per step = exp(-ln(2) / half_life)
        decay_factor = np.exp(-np.log(2) / half_life_periods)
        
        enriched_df = df.copy()

        for col in event_columns:
            if col not in df.columns:
                logger.warning(f"Event column '{col}' not found in DataFrame. Skipping.")
                continue

            decayed_values = np.zeros(len(df))
            current_value = 0.0

            # Use a simple loop to apply decay with resets
            # Resetting to 1.0 on new event (rather than stacking) to prevent extreme outliers for ML models
            for i in range(len(df)):
                if df[col].iloc[i] >= 1:
                    current_value = 1.0
                else:
                    current_value *= decay_factor
                
                decayed_values[i] = current_value

            enriched_df[f"{col}_decayed"] = decayed_values
            logger.debug(f"Added decay feature for '{col}' with half-life {half_life_periods}.")

        return enriched_df