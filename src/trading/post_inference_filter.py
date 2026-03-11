"""
Post-Inference Filter to adjust model prediction confidence based on market context.

This module takes raw model predictions and multiplies their confidence scores
by a weighted average of contextual multipliers (e.g., macro, RSI, sentiment).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.core.logging.logger import ProjectLogger

class PostInferenceFilter:
    """
    Applies contextual filters to model predictions to refine confidence scores.
    All operations are vectorized for efficiency.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the filter with configurable weights and thresholds.

        Args:
            config (dict, optional): A dictionary with filter parameters.
        """
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        if config is None:
            config = {}
        
        self.params = {
            'macro_weight': config.get('macro_weight', 0.3),
            'rsi_weight': config.get('rsi_weight', 0.2),
            'sentiment_weight': config.get('sentiment_weight', 0.4),
            'min_confidence': config.get('min_confidence', 0.1),
            'max_confidence': config.get('max_confidence', 0.95)
        }
        self.logger.info(f"PostInferenceFilter initialized with weights: {self.params}")

    def _get_macro_multiplier(self, macro_strength: pd.Series) -> pd.Series:
        """Calculates confidence multiplier based on a decaying macro signal."""
        # A weak or non-existent signal provides a neutral or slightly negative multiplier.
        # A strong signal (close to 1.0) provides a positive boost.
        multiplier = 0.5 + macro_strength
        return multiplier.fillna(0.5).clip(0.5, 1.5)

    def _get_rsi_multiplier(self, rsi_series: pd.Series) -> pd.Series:
        """Calculates confidence multiplier based on RSI extreme levels."""
        conditions = [
            (rsi_series > 70) | (rsi_series < 30), # Extreme levels -> High confidence
            (rsi_series > 60) | (rsi_series < 40)  # Moderate levels -> Mild confidence
        ]
        choices = [1.2, 1.1] # Boosts
        # Use np.select for vectorized conditional logic
        return pd.Series(np.select(conditions, choices, default=0.9), index=rsi_series.index)

    def _get_sentiment_multiplier(self, sentiment_score: pd.Series) -> pd.Series:
        """Calculates confidence multiplier based on sentiment strength."""
        abs_sentiment = sentiment_score.abs()
        conditions = [
            abs_sentiment > 0.8, # Very strong sentiment
            abs_sentiment > 0.5, # Moderate sentiment
            abs_sentiment > 0.2  # Weak sentiment
        ]
        choices = [1.2, 1.1, 1.0]
        return pd.Series(np.select(conditions, choices, default=0.9), index=sentiment_score.index)

    def apply(self, 
              predictions_df: pd.DataFrame,
              confidence_col: str = 'confidence',
              macro_col: Optional[str] = 'macro_decayed_strength',
              rsi_col: Optional[str] = 'RSI_14',
              sentiment_col: Optional[str] = 'sentiment_score') -> pd.DataFrame:
        """
        Applies the vectorized post-inference filter to a DataFrame of predictions.

        Args:
            predictions_df: DataFrame with model predictions and contextual features.
            confidence_col: The name of the column holding the original prediction confidence.
            macro_col: Column name for the macro signal strength.
            rsi_col: Column name for the RSI indicator.
            sentiment_col: Column name for the sentiment score.

        Returns:
            pd.DataFrame: The DataFrame with added columns for filtered confidence.
        """
        self.logger.info(f"Applying post-inference filter to {len(predictions_df)} predictions...")
        
        if confidence_col not in predictions_df.columns:
            raise ValueError(f"Confidence column '{confidence_col}' not found in DataFrame.")

        result_df = predictions_df.copy()
        result_df['original_confidence'] = result_df[confidence_col]
        
        # --- Calculate all multipliers vectorially ---
        multipliers = pd.DataFrame(index=result_df.index)
        total_weight = 0

        if macro_col and macro_col in result_df.columns:
            multipliers['macro'] = self._get_macro_multiplier(result_df[macro_col])
            total_weight += self.params['macro_weight']
        else:
             multipliers['macro'] = 1.0

        if rsi_col and rsi_col in result_df.columns:
            multipliers['rsi'] = self._get_rsi_multiplier(result_df[rsi_col])
            total_weight += self.params['rsi_weight']
        else:
            multipliers['rsi'] = 1.0

        if sentiment_col and sentiment_col in result_df.columns:
            multipliers['sentiment'] = self._get_sentiment_multiplier(result_df[sentiment_col])
            total_weight += self.params['sentiment_weight']
        else:
            multipliers['sentiment'] = 1.0

        # --- Calculate the weighted average multiplier ---
        weighted_multiplier = (
            multipliers['macro'] * self.params['macro_weight'] +
            multipliers['rsi'] * self.params['rsi_weight'] +
            multipliers['sentiment'] * self.params['sentiment_weight']
        ) / total_weight

        result_df['confidence_multiplier'] = weighted_multiplier
        result_df['filtered_confidence'] = result_df['original_confidence'] * weighted_multiplier

        # --- Clip to final min/max bounds ---
        result_df['filtered_confidence'] = result_df['filtered_confidence'].clip(
            self.params['min_confidence'], self.params['max_confidence']
        )

        # --- Logging Statistics ---
        avg_original = result_df['original_confidence'].mean()
        avg_filtered = result_df['filtered_confidence'].mean()
        self.logger.info("Filter application complete.")
        self.logger.info(f"Average Confidence: Original={avg_original:.3f}, Filtered={avg_filtered:.3f}")

        return result_df
