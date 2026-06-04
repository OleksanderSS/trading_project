import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class SentimentStatsCalculator:
    """
    Provides vectorized methods for calculating statistics on sentiment data.
    """

    @staticmethod
    def calculate_sentiment_stats(news_data: pd.DataFrame, column: str = 'news_score') -> Dict[str, float]:
        """
        Calculates mean, standard deviation, and dynamic thresholds for a sentiment score series.

        Args:
            news_data (pd.DataFrame): DataFrame containing the news data.
            column (str): The name of the column with sentiment scores.

        Returns:
            Dict[str, float]: A dictionary with statistics including 'mean', 'std', 
                              'positive_threshold', and 'negative_threshold'.
        """
        # Validate input data and column
        is_data_invalid = news_data is None or (isinstance(news_data, pd.DataFrame) and news_data.empty)
        is_column_missing = news_data is not None and column not in news_data.columns
        
        if is_data_invalid or is_column_missing:
            logger.warning(f"Cannot calculate sentiment stats. Input data is empty or column '{column}' is missing.")
            return {"mean": 0.0, "std": 0.0, "positive_threshold": 0.5, "negative_threshold": -0.5}

        scores = news_data[column].dropna()
        
        if scores.empty:
            return {"mean": 0.0, "std": 0.0, "positive_threshold": 0.5, "negative_threshold": -0.5}

        mean_score = float(scores.mean())
        std_score = float(scores.std())

        if pd.isna(std_score):
            std_score = 0.0

        positive_threshold = mean_score + std_score
        negative_threshold = mean_score - std_score

        logger.info(f"Calculated sentiment stats for column '{column}': mean={mean_score:.3f}, std={std_score:.3f}")

        return {
            "mean": mean_score,
            "std": std_score,
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold
        }
