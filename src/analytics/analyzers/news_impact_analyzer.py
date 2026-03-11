from ..interfaces import IAnalyzer
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.core.logging.logger import ProjectLogger
from src.sentiment.sentiment_models import analyze_sentiment

logger = ProjectLogger.get_logger(__name__)

class NewsImpactAnalyzer(IAnalyzer):
    """
    Analyzes raw news text to calculate a sentiment-based, time-decaying impact score.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the analyzer with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary which can contain:
                - sentiment_weights (dict): Weights for sentiment labels (e.g., {'positive': 1, 'negative': -1.5}).
                - half_life_hours (int): Half-life for the score's exponential decay.
        """
        self.config = config or {}
        # Provide default weights if not specified
        self.sentiment_weights = self.config.get('sentiment_weights', {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        })
        self.half_life_hours = self.config.get('half_life_hours', 48)
        logger.info(f"NewsImpactAnalyzer initialized with half-life={self.half_life_hours}h and weights={self.sentiment_weights}")

    def _calculate_decay_factor(self, series_freq_hours: float) -> float:
        """
        Calculates the exponential decay factor based on data frequency.
        """
        if self.half_life_hours <= 0:
            return 0.0  # A factor of 0 means no decay (alpha=1 in ewm)
        return np.exp(-np.log(2) * (series_freq_hours / self.half_life_hours))

    def analyze(self, news_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Performs sentiment analysis and calculates a weighted, time-decaying news impact score.

        Args:
            news_data (pd.DataFrame): DataFrame with a 'timestamp' index and a 'text' column containing news content.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'news_impact_scores': pd.Series of the calculated scores.
                            - 'news_significance_levels': pd.Series of the significance levels.
        """
        if not isinstance(news_data, pd.DataFrame) or news_data.empty or 'text' not in news_data.columns:
            logger.warning("Input data is not a valid DataFrame, is empty, or lacks a 'text' column. Skipping analysis.")
            return {}

        # 1. Perform Sentiment Analysis
        logger.info(f"Starting sentiment analysis for {len(news_data)} news items...")
        # Assuming news_data index is the timestamp
        sentiment_results = analyze_sentiment(news_data['text'].tolist())
        if sentiment_results.empty:
            logger.warning("Sentiment analysis returned no results.")
            return {}
            
        # Use original index from news_data for alignment
        sentiment_results.index = news_data.index

        # 2. Calculate a single weighted sentiment score for each news item
        sentiment_results['weighted_score'] = sentiment_results.apply(
            lambda row: row['score'] * self.sentiment_weights.get(row['label'], 0),
            axis=1
        )

        # 3. Aggregate scores per timestamp (if multiple news items have the same timestamp)
        # Resample to the original frequency to ensure all time points are kept
        if len(news_data.index) > 1:
            inferred_freq = pd.infer_freq(news_data.index)
        else:
            inferred_freq = None # Cannot infer with one point

        if inferred_freq:
            aggregated_scores = sentiment_results['weighted_score'].resample(inferred_freq).sum()
        else:
            # If frequency cannot be inferred, group by timestamp
            aggregated_scores = sentiment_results['weighted_score'].groupby(sentiment_results.index).sum()


        # 4. Apply Time-Decaying EMA
        if len(aggregated_scores.index) < 2:
            series_freq_hours = 0.0 # Cannot determine frequency
        else:
            median_diff = aggregated_scores.index.to_series().diff().median()
            series_freq_hours = pd.to_timedelta(median_diff).total_seconds() / 3600
        
        decay_factor = self._calculate_decay_factor(series_freq_hours)

        if decay_factor > 0:
            # alpha = 1 - decay_factor. An alpha of 1 means no smoothing.
            impact_score_series = aggregated_scores.ewm(alpha=1-decay_factor, adjust=False).mean()
        else:
            impact_score_series = aggregated_scores
        
        # 5. Determine significance level
        significance_thresholds = self.config.get('significance_thresholds', {})
        high_impact_threshold = significance_thresholds.get('high_impact', 0.8)
        medium_impact_threshold = significance_thresholds.get('medium_impact', 0.3)

        def get_significance(score):
            abs_score = abs(score)
            if abs_score >= high_impact_threshold:
                return 'high'
            elif abs_score >= medium_impact_threshold:
                return 'medium'
            else:
                return 'low'

        significance_series = impact_score_series.apply(get_significance).astype("category")

        logger.info("Successfully calculated news impact and significance scores.")

        return {
            'news_impact_scores': impact_score_series,
            'news_significance_levels': significance_series
        }
