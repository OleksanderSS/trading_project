"""
Post-Inference Filter to adjust model prediction confidence based on market context.

This module takes raw model predictions and multiplies their confidence scores
by a weighted average of contextual multipliers (e.g., macro, RSI, sentiment, and CHAOS).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.core.logging.logger import ProjectLogger

class PostInferenceFilter:
    """
    Applies contextual filters to model predictions to refine confidence scores.
    ✅ PATTERN & CHAOS AWARE:
    - Penalizes confidence during high Context Velocity.
    - Boosts confidence for stable, high-reliability patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the filter with configurable weights and thresholds.
        """
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        if config is None:
            config = {}
        
        self.params = {
            'macro_weight': config.get('macro_weight', 0.2),
            'rsi_weight': config.get('rsi_weight', 0.1),
            'sentiment_weight': config.get('sentiment_weight', 0.3),
            'chaos_weight': config.get('chaos_weight', 0.4), # ✅ ELITE: Chaos has highest weight
            'min_confidence': config.get('min_confidence', 0.0), # Allow zero confidence for kill-switch
            'max_confidence': config.get('max_confidence', 0.95)
        }
        self.logger.info(f"PostInferenceFilter initialized with Pattern & Chaos support.")

    def _get_chaos_multiplier(self, velocity: pd.Series) -> pd.Series:
        """
        🎯 ANXIETY PENALTY:
        Calculates multiplier based on Context Velocity (0.0 to 1.0).
        High velocity = High chaos = Low confidence.
        """
        conditions = [
            velocity > 0.85, # Extreme chaos -> Kill-switch proxy
            velocity > 0.70, # High chaos -> Heavy penalty
            velocity > 0.40, # Moderate chaos -> Light penalty
            velocity <= 0.20 # High stability -> Small boost
        ]
        choices = [0.1, 0.5, 0.8, 1.2]
        return pd.Series(np.select(conditions, choices, default=1.0), index=velocity.index)

    def _get_macro_multiplier(self, macro_strength: pd.Series) -> pd.Series:
        multiplier = 0.5 + macro_strength
        return multiplier.where(multiplier.notna(), 0.5).clip(0.5, 1.5)

    def _get_rsi_multiplier(self, rsi_series: pd.Series) -> pd.Series:
        conditions = [
            (rsi_series > 70) | (rsi_series < 30),
            (rsi_series > 60) | (rsi_series < 40)
        ]
        choices = [1.2, 1.1]
        return pd.Series(np.select(conditions, choices, default=0.9), index=rsi_series.index)

    def _get_sentiment_multiplier(self, sentiment_score: pd.Series) -> pd.Series:
        abs_sentiment = sentiment_score.abs()
        conditions = [
            abs_sentiment > 0.8,
            abs_sentiment > 0.5,
            abs_sentiment > 0.2
        ]
        choices = [1.2, 1.1, 1.0]
        return pd.Series(np.select(conditions, choices, default=0.9), index=sentiment_score.index)

    def apply(self, predictions_df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Applies vectorized filters including Chaos and Pattern awareness.
        """
        if config is None: config = {}
        
        confidence_col = config.get('confidence_col', 'confidence')
        macro_col = config.get('macro_col', 'macro_decayed_strength')
        rsi_col = config.get('rsi_col', 'RSI_14')
        sentiment_col = config.get('sentiment_col', 'sentiment_score')
        velocity_col = config.get('velocity_col', 'context_velocity')
        
        self._validate_input(predictions_df, confidence_col)
        result_df = predictions_df.copy()
        result_df['original_confidence'] = result_df[confidence_col]
        
        # 1. Calculate Multipliers
        multipliers = pd.DataFrame(index=result_df.index)
        
        # Macro
        if macro_col in result_df.columns:
            multipliers['macro'] = self._get_macro_multiplier(result_df[macro_col])
        else: multipliers['macro'] = 1.0
            
        # RSI
        if rsi_col in result_df.columns:
            multipliers['rsi'] = self._get_rsi_multiplier(result_df[rsi_col])
        else: multipliers['rsi'] = 1.0
            
        # Sentiment
        if sentiment_col in result_df.columns:
            multipliers['sentiment'] = self._get_sentiment_multiplier(result_df[sentiment_col])
        else: multipliers['sentiment'] = 1.0
            
        # ✅ ELITE: Chaos (Velocity)
        if velocity_col in result_df.columns:
            multipliers['chaos'] = self._get_chaos_multiplier(result_df[velocity_col])
            self.logger.info(f"🛡️ Applied Chaos-Aware penalty using {velocity_col}")
        else: multipliers['chaos'] = 1.0

        # 2. Weighted Synthesis
        weights = self.params
        weighted_multiplier = (
            multipliers['macro'] * weights['macro_weight'] +
            multipliers['rsi'] * weights['rsi_weight'] +
            multipliers['sentiment'] * weights['sentiment_weight'] +
            multipliers['chaos'] * weights['chaos_weight']
        )
        
        result_df['confidence_multiplier'] = weighted_multiplier
        result_df['filtered_confidence'] = result_df['original_confidence'] * weighted_multiplier
        
        # 3. Apply bounds
        result_df['filtered_confidence'] = result_df['filtered_confidence'].clip(
            self.params['min_confidence'], self.params['max_confidence']
        )
        
        return result_df
    
    def _validate_input(self, predictions_df: pd.DataFrame, confidence_col: str):
        if confidence_col not in predictions_df.columns:
             # If missing, we add it with a default to allow filtering
             predictions_df[confidence_col] = 0.5
