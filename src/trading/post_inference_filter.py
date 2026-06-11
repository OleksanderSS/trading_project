"""
Post-Inference Filter to adjust model prediction confidence based on market context.

This module takes raw model predictions and multiplies their confidence scores
by a weighted average of contextual multipliers (e.g., macro, RSI, sentiment).
"""


import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


class PostInferenceFilter:
    """
    Applies contextual filters to model predictions to refine confidence scores.
    All operations are vectorized for efficiency.
    """

    def __init__(self, config: dict | None = None):
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

    def apply(self, predictions_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
        """
        Applies vectorized post-inference filter to a DataFrame of predictions.

        Args:
            predictions_df: DataFrame with model predictions and contextual features.
            config: Configuration dictionary with column names:
                - confidence_col: The name of column holding the original prediction confidence
                - macro_col: Column name for macro signal strength
                - rsi_col: Column name for RSI indicator
                - sentiment_col: Column name for sentiment score

        Returns:
            pd.DataFrame: The DataFrame with added columns for filtered confidence.
        """
        # Set default configuration
        if config is None:
            config = {}

        confidence_col = config.get('confidence_col', 'confidence')
        macro_col = config.get('macro_col', 'macro_decayed_strength')
        rsi_col = config.get('rsi_col', 'RSI_14')
        sentiment_col = config.get('sentiment_col', 'sentiment_score')

        self.logger.info(f"Applying post-inference filter to {len(predictions_df)} predictions...")

        self._validate_input(predictions_df, confidence_col)
        result_df = self._prepare_result_dataframe(predictions_df, confidence_col)

        multipliers, total_weight, active_weights = self._calculate_multipliers(result_df, macro_col, rsi_col, sentiment_col)
        weighted_multiplier = self._calculate_weighted_multiplier(multipliers, total_weight, active_weights)

        result_df = self._apply_final_adjustments(result_df, weighted_multiplier)
        self._log_statistics(result_df)

        return result_df

    def _validate_input(self, predictions_df: pd.DataFrame, confidence_col: str):
        """Validate input DataFrame and required columns."""
        if confidence_col not in predictions_df.columns:
            raise ValueError(f"Confidence column '{confidence_col}' not found in DataFrame.")

    def _prepare_result_dataframe(self, predictions_df: pd.DataFrame, confidence_col: str) -> pd.DataFrame:
        """Prepare result DataFrame with original confidence column."""
        result_df = predictions_df.copy()
        result_df['original_confidence'] = result_df[confidence_col]
        return result_df

    def _calculate_multipliers(self, result_df: pd.DataFrame, macro_col: str, rsi_col: str, sentiment_col: str) -> tuple:
        """Calculate all multipliers vectorially."""
        multipliers = pd.DataFrame(index=result_df.index)
        total_weight = 0
        active_weights = {}

        macro_multiplier, macro_weight = self._get_macro_data(result_df, macro_col)
        multipliers['macro'] = macro_multiplier
        total_weight += macro_weight
        active_weights['macro'] = macro_weight

        rsi_multiplier, rsi_weight = self._get_rsi_data(result_df, rsi_col)
        multipliers['rsi'] = rsi_multiplier
        total_weight += rsi_weight
        active_weights['rsi'] = rsi_weight

        sentiment_multiplier, sentiment_weight = self._get_sentiment_data(result_df, sentiment_col)
        multipliers['sentiment'] = sentiment_multiplier
        total_weight += sentiment_weight
        active_weights['sentiment'] = sentiment_weight

        return multipliers, total_weight, active_weights

    def _get_macro_data(self, result_df: pd.DataFrame, macro_col: str) -> tuple:
        """Get macro multiplier and weight."""
        if macro_col and macro_col in result_df.columns:
            return self._get_macro_multiplier(result_df[macro_col]), self.params['macro_weight']
        return 1.0, 0

    def _get_rsi_data(self, result_df: pd.DataFrame, rsi_col: str) -> tuple:
        """Get RSI multiplier and weight."""
        if rsi_col and rsi_col in result_df.columns:
            return self._get_rsi_multiplier(result_df[rsi_col]), self.params['rsi_weight']
        return 1.0, 0

    def _get_sentiment_data(self, result_df: pd.DataFrame, sentiment_col: str) -> tuple:
        """Get sentiment multiplier and weight."""
        if sentiment_col and sentiment_col in result_df.columns:
            return self._get_sentiment_multiplier(result_df[sentiment_col]), self.params['sentiment_weight']
        return 1.0, 0

    def _calculate_weighted_multiplier(self, multipliers: pd.DataFrame, total_weight: float, active_weights: dict) -> pd.Series:
        """Calculate weighted average multiplier, ignoring inactive parameters to prevent inflation."""
        if total_weight <= 0:
            return pd.Series(1.0, index=multipliers.index)
        return (
            multipliers['macro'] * active_weights['macro'] +
            multipliers['rsi'] * active_weights['rsi'] +
            multipliers['sentiment'] * active_weights['sentiment']
        ) / total_weight

    def _apply_final_adjustments(self, result_df: pd.DataFrame, weighted_multiplier: pd.Series) -> pd.DataFrame:
        """Apply final adjustments to result DataFrame."""
        result_df['confidence_multiplier'] = weighted_multiplier
        result_df['filtered_confidence'] = result_df['original_confidence'] * weighted_multiplier

        # Clip to final min/max bounds
        result_df['filtered_confidence'] = result_df['filtered_confidence'].clip(
            self.params['min_confidence'], self.params['max_confidence']
        )

        return result_df

    def _log_statistics(self, result_df: pd.DataFrame):
        """Log filtering statistics."""
        avg_original = result_df['original_confidence'].mean()
        avg_filtered = result_df['filtered_confidence'].mean()
        self.logger.info("Filter application complete.")
        self.logger.info(f"Average Confidence: Original={avg_original:.3f}, Filtered={avg_filtered:.3f}")
