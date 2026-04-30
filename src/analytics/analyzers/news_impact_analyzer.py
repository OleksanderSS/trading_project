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
        if not self._validate_input_data(news_data):
            return {}

        sentiment_results = self._perform_sentiment_analysis(news_data)
        if sentiment_results.empty:
            return {}

        weighted_scores = self._calculate_weighted_scores(sentiment_results)
        aggregated_scores = self._aggregate_scores_by_timestamp(weighted_scores, news_data)
        impact_scores = self._apply_time_decay(aggregated_scores)
        significance_levels = self._determine_significance_levels(impact_scores)

        logger.info("Successfully calculated news impact and significance scores.")

        return {
            'news_impact_scores': impact_scores,
            'news_significance_levels': significance_levels
        }

    def _validate_input_data(self, news_data: pd.DataFrame) -> bool:
        """Validate input data for sentiment analysis."""
        if not isinstance(news_data, pd.DataFrame) or news_data.empty or 'text' not in news_data.columns:
            logger.warning("Input data is not a valid DataFrame, is empty, or lacks a 'text' column. Skipping analysis.")
            return False
        return True

    def _perform_sentiment_analysis(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis on news data."""
        logger.info(f"Starting sentiment analysis for {len(news_data)} news items...")
        sentiment_results = analyze_sentiment(news_data['text'].tolist())
        if sentiment_results.empty:
            logger.warning("Sentiment analysis returned no results.")
            return pd.DataFrame()
        
        # Use original index from news_data for alignment
        sentiment_results.index = news_data.index
        return sentiment_results

    def _calculate_weighted_scores(self, sentiment_results: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted sentiment scores."""
        sentiment_results['weighted_score'] = sentiment_results.apply(
            lambda row: row['score'] * self.sentiment_weights.get(row['label'], 0),
            axis=1
        )
        return sentiment_results

    def _aggregate_scores_by_timestamp(self, sentiment_results: pd.DataFrame, news_data: pd.DataFrame) -> pd.Series:
        """Aggregate scores per timestamp."""
        try:
            if len(news_data.index) > 1:
                inferred_freq = pd.infer_freq(news_data.index)
            else:
                inferred_freq = None

            if inferred_freq:
                return sentiment_results['weighted_score'].resample(inferred_freq).sum()
            else:
                logger.warning("Cannot infer frequency from sparse data, grouping by timestamp instead")
                return sentiment_results['weighted_score'].groupby(sentiment_results.index).sum()
        except Exception as e:
            logger.warning(f"Error inferring frequency: {e}, falling back to timestamp grouping")
            return sentiment_results['weighted_score'].groupby(sentiment_results.index).sum()

    def _apply_time_decay(self, aggregated_scores: pd.Series) -> pd.Series:
        """Apply time-decaying EMA to scores."""
        if len(aggregated_scores.index) < 2:
            series_freq_hours = 0.0
        else:
            median_diff = aggregated_scores.index.to_series().diff().median()
            series_freq_hours = pd.to_timedelta(median_diff).total_seconds() / 3600
        
        decay_factor = self._calculate_decay_factor(series_freq_hours)

        if decay_factor > 0:
            return aggregated_scores.ewm(alpha=1-decay_factor, adjust=False).mean()
        else:
            return aggregated_scores

    def _determine_significance_levels(self, impact_scores: pd.Series) -> pd.Series:
        """Determine significance levels for impact scores."""
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

        return impact_scores.apply(get_significance).astype("category")
