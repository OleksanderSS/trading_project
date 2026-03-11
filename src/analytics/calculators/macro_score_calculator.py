import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MacroScoreCalculator:
    """
    Calculates a composite macroeconomic score from various weighted indicators.
    This score represents a single, normalized value for the overall macro environment.
    """

    def __init__(self, indicators_config: Dict[str, Dict]):
        """
        Initializes the calculator with configuration for macro indicators.

        Args:
            indicators_config (Dict[str, Dict]): A dictionary where keys are indicator names
                                                 and values are their configurations (weight, direction).
        """
        if not indicators_config:
            raise ValueError("Indicators configuration cannot be empty.")
        self.indicators_config = indicators_config
        logger.info(f"MacroScoreCalculator initialized with indicators: {list(self.indicators_config.keys())}")

    def calculate_composite_score(self, macro_data: pd.DataFrame, rolling_window: int = 252) -> pd.DataFrame:
        """
        Calculates the composite macro score over a rolling window.

        Args:
            macro_data (pd.DataFrame): DataFrame with macro indicators as columns.
            rolling_window (int): The window size for rolling normalization (e.g., 252 for 1 year).

        Returns:
            pd.DataFrame: A DataFrame with the composite macro score and its components.
        """
        if not isinstance(macro_data, pd.DataFrame) or macro_data.empty:
            logger.warning("Macro data is empty or not a DataFrame. Returning empty DataFrame.")
            return pd.DataFrame()

        all_scores = {}
        for indicator, config in self.indicators_config.items():
            if indicator not in macro_data.columns:
                logger.warning(f"Indicator '{indicator}' not found in macro data. Skipping.")
                continue

            series = macro_data[indicator].dropna()
            
            # 1. Transform: Use percentage change to represent momentum
            transformed_series = series.pct_change(periods=int(rolling_window/12)).fillna(0) # Monthly change assumption
            
            # 2. Normalize: Rolling Z-score to standardize the data
            mean = transformed_series.rolling(window=rolling_window, min_periods=int(rolling_window*0.8)).mean()
            std = transformed_series.rolling(window=rolling_window, min_periods=int(rolling_window*0.8)).std()
            normalized_series = (transformed_series - mean) / std.replace(0, 1)

            # 3. Directional Alignment
            if config.get('direction', 'positive') == 'negative':
                normalized_series = -normalized_series

            all_scores[f"{indicator}_score"] = normalized_series.fillna(0)

        if not all_scores:
            logger.error("No indicators were processed. Cannot calculate composite score.")
            return pd.DataFrame(index=macro_data.index).assign(composite_macro_score=0.0)

        scores_df = pd.DataFrame(all_scores, index=macro_data.index)
        
        # 4. Weighting and Aggregation
        composite_score = pd.Series(0.0, index=scores_df.index)
        total_weight = sum(config['weight'] for config in self.indicators_config.values())
        
        if total_weight == 0: 
            logger.warning("Total weight of indicators is zero. Composite score will be zero.")
            return scores_df.assign(composite_macro_score=0.0)

        for indicator, config in self.indicators_config.items():
            score_col = f"{indicator}_score"
            if score_col in scores_df.columns:
                composite_score += scores_df[score_col] * (config['weight'] / total_weight)

        # 5. Final Scaling: Scale to a consistent range (e.g., 0 to 100)
        final_composite_score = pd.Series(minmax_scale(composite_score.fillna(0), feature_range=(0, 100)), index=composite_score.index)
        
        scores_df['composite_macro_score'] = final_composite_score
        logger.info("Successfully calculated the composite macro score.")
        
        return scores_df
