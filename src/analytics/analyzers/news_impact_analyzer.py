from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.sentiment.sentiment_models import analyze_sentiment

from ..interfaces import IAnalyzer

logger = ProjectLogger.get_logger(__name__)


class NewsImpactAnalyzer(IAnalyzer):
    """
    Analyzes raw news text to calculate a sentiment-based, time-decaying impact score.
    """

    def __init__(self, config: dict[str, Any] | None=None):
        """
        Initializes the analyzer with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary which can contain:
                - sentiment_weights (dict): Weights for sentiment labels (e.g., {'positive': 1, 'negative': -1.5}).
                - half_life_hours (int): Half-life for the score's exponential decay.
        """
        self.config = config or {}
        self.sentiment_weights = self.config.get('sentiment_weights', {
            'positive': 1.0, 'negative': -1.0, 'neutral': 0.0})
        self.half_life_hours = self.config.get('half_life_hours', 48)
        logger.info(
            f'NewsImpactAnalyzer initialized with half-life={self.half_life_hours}h and weights={self.sentiment_weights}'
            )

    def _calculate_decay_factor(self, series_freq_hours: float) ->float:
        """
        Calculates the exponential decay factor based on data frequency.
        """
        if self.half_life_hours <= 0:
            return 0.0
        return np.exp(-np.log(2) * (series_freq_hours / self.half_life_hours))

    def analyze(self, news_data: pd.DataFrame, **kwargs) ->dict[str, Any]:
        """
        Performs sentiment analysis and calculates a weighted, time-decaying news impact score.
        """
        if not isinstance(news_data, pd.DataFrame) or news_data.empty or 'text' not in news_data.columns:
            raise DataProcessingError("Input data must be a non-empty DataFrame with a 'text' column.")

        sentiment_results = self._perform_sentiment_analysis(news_data)

        weighted_scores = self._calculate_weighted_scores(sentiment_results)
        aggregated_scores = self._aggregate_scores_by_timestamp(weighted_scores
            , news_data)
        impact_scores = self._apply_time_decay(aggregated_scores)
        significance_levels = self._determine_significance_levels(impact_scores
            )
        logger.info(
            'Successfully calculated news impact and significance scores.')
        return {'news_impact_scores': impact_scores,
            'news_significance_levels': significance_levels}

    def _perform_sentiment_analysis(self, news_data: pd.DataFrame
        ) ->pd.DataFrame:
        """Perform sentiment analysis on news data."""
        logger.info(
            f'Starting sentiment analysis for {len(news_data)} news items...')
        sentiment_results = analyze_sentiment(news_data['text'].tolist())
        if sentiment_results.empty:
            raise DataProcessingError('Sentiment analysis returned no results.')

        sentiment_results.index = news_data.index
        return sentiment_results

    def _calculate_weighted_scores(self, sentiment_results: pd.DataFrame
        ) ->pd.DataFrame:
        """Calculate weighted sentiment scores."""
        sentiment_results['weighted_score'] = sentiment_results.apply(lambda
            row: row['score'] * self.sentiment_weights.get(row['label'], 0),
            axis=1)
        return sentiment_results

    def _aggregate_scores_by_timestamp(self, sentiment_results: pd.
        DataFrame, news_data: pd.DataFrame) ->pd.Series:
        """Aggregate scores per timestamp."""
        try:
            if len(news_data.index) > 1:
                inferred_freq = pd.infer_freq(news_data.index)
            else:
                inferred_freq = None
            if inferred_freq:
                return sentiment_results['weighted_score'].resample(
                    inferred_freq).sum()
            else:
                logger.warning(
                    'Cannot infer frequency from sparse data, grouping by timestamp instead'
                    )
                return sentiment_results['weighted_score'].groupby(
                    sentiment_results.index).sum()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise DataProcessingError(f'Error aggregating scores by timestamp: {e}') from e

    def _apply_time_decay(self, aggregated_scores: pd.Series) ->pd.Series:
        """Apply time-decaying EMA to scores."""
        if len(aggregated_scores.index) < 2:
            series_freq_hours = 0.0
        else:
            median_diff = aggregated_scores.index.to_series().diff().median()
            series_freq_hours = pd.to_timedelta(median_diff).total_seconds(
                ) / 3600
        decay_factor = self._calculate_decay_factor(series_freq_hours)
        if decay_factor > 0:
            return aggregated_scores.ewm(alpha=1 - decay_factor, adjust=False
                ).mean()
        else:
            return aggregated_scores

    def _determine_significance_levels(self, impact_scores: pd.Series
        ) ->pd.Series:
        """How unusual this impact is, against the impacts seen before it.

        The thresholds were absolute -- 0.8 for high, 0.3 for medium -- and
        the score never approaches them. Measured on the 2026-08-15 batch it
        spans 0.058 to 0.102 in magnitude, so every bar was 'low' and the
        feature reached the models as the constant 0.

        Absolute cut-offs cannot work here whatever numbers are chosen. The
        impact score is a time-decayed sum of weighted sentiment, so its scale
        follows how many articles were collected in the window: extending the
        news history, adding a source or changing the half-life all move it.
        A threshold calibrated today would be wrong after the next collection.

        So significance is now RELATIVE -- a score is 'high' if it exceeds the
        90th percentile of the magnitudes seen up to and including it, and
        'medium' above the 70th. `expanding` is what makes that honest: at row
        i it reads rows 0..i and never the future, so the label a bar carries
        could have been computed at that bar. A whole-series `quantile()` would
        have been simpler and would have leaked the distribution backwards.

        Until there is enough history to say what "usual" is
        (`min_history`, default 20 observations) everything is 'low', because
        an unusual value cannot be recognised without a baseline. That is a
        statement of ignorance, not a measurement of insignificance.

        Absolute thresholds remain available: set `significance_mode: absolute`
        with `high_impact` / `medium_impact` for the old behaviour.
        """
        cfg = self.config.get('significance_thresholds', {}) or {}
        magnitude = impact_scores.abs()

        if str(cfg.get('significance_mode', 'relative')).lower() == 'absolute':
            high = float(cfg.get('high_impact', 0.8))
            medium = float(cfg.get('medium_impact', 0.3))
            levels = pd.Series(
                np.where(magnitude >= high, 'high',
                         np.where(magnitude >= medium, 'medium', 'low')),
                index=impact_scores.index,
            )
            if levels.nunique() <= 1:
                logger.error(
                    "Absolute significance thresholds (%.3f / %.3f) produced a "
                    "single level across %d scores spanning %.4f to %.4f. The "
                    "feature is a constant and cannot inform anything.",
                    high, medium, len(magnitude),
                    float(magnitude.min()), float(magnitude.max()),
                )
            return levels.astype('category')

        high_q = float(cfg.get('high_quantile', 0.90))
        medium_q = float(cfg.get('medium_quantile', 0.70))
        min_history = int(cfg.get('min_history', 20))

        expanding = magnitude.expanding(min_periods=min_history)
        high_cut = expanding.quantile(high_q)
        medium_cut = expanding.quantile(medium_q)

        levels = pd.Series('low', index=impact_scores.index, dtype=object)
        levels[magnitude >= medium_cut] = 'medium'
        levels[magnitude >= high_cut] = 'high'
        return levels.astype('category')
