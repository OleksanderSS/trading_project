# src/features/nlp/scoring/news_scorer.py

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_SCORING_CONFIG = {
    # Defines the influence of each component on the final score's MAGNITUDE.
    'weights': {
        'relevance': 0.6,  # Contribution from keywords
        'significance': 0.4  # Contribution from entities (e.g., primary tickers)
    },
    # Parameters to tune the behavior of each scoring component.
    'parameters': {
        'keyword_cap': 10,  # After this many keywords, the score contribution maxes out.
        'primary_ticker_bonus': 1.0, # Score for finding a primary ticker.
        'other_entity_bonus': 0.2  # Score for finding any other relevant entity.
    }
}

class NewsScorer:
    """
    Calculates a sophisticated score for news articles by combining sentiment,
    keyword relevance, and entity significance.

    The final score's direction is determined by sentiment, and its magnitude
    is amplified by relevance (keywords) and significance (entities).
    """

    def __init__(self, scorer_config: dict[str, Any] | None = None, primary_tickers: list[str] | None = None):
        """
        Initializes the NewsScorer.

        Args:
            scorer_config (Optional[Dict[str, Any]]): Configuration for weights and parameters.
                                                        If None, uses default values.
            primary_tickers (Optional[List[str]]): A list of tickers to treat as highly significant.
        """
        config = scorer_config or DEFAULT_SCORING_CONFIG
        self.weights = config.get('weights', DEFAULT_SCORING_CONFIG['weights'])
        self.params = config.get('parameters', DEFAULT_SCORING_CONFIG['parameters'])
        self.primary_tickers: set[str] = {ticker.upper() for ticker in primary_tickers} if primary_tickers else set()

        logger.info(f"NewsScorer initialized with primary tickers: {self.primary_tickers or 'None'}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Scorer weights: {self.weights}, parameters: {self.params}")

    def _calculate_directional_sentiment(self, sentiment_details: dict[str, float]) -> float:
        """
        Calculates a directional sentiment score from -1.0 (very negative) to +1.0 (very positive).
        This is based on the difference between positive and negative sentiment components.
        """
        if not sentiment_details:
            return 0.0

        # Assumes details contain 'positive' and 'negative' scores.
        pos_score = sentiment_details.get('positive', 0.0)
        neg_score = sentiment_details.get('negative', 0.0)

        return pos_score - neg_score

    def _calculate_relevance_score(self, keywords: list[str]) -> float:
        """
        Calculates a relevance score based on the number of keywords found.
        Uses a logarithmic scale to give diminishing returns for additional keywords.
        Result is normalized between 0.0 and 1.0.
        """
        if not keywords:
            return 0.0

        cap = self.params.get('keyword_cap', 10)
        # Using log1p for a smooth curve where the first few keywords matter most.
        # log1p(x) is log(1+x), avoiding math errors for 0 keywords.
        normalized_score = math.log1p(len(keywords)) / math.log1p(cap)

        return min(normalized_score, 1.0) # Ensure score does not exceed 1.0

    def _calculate_significance_score(self, entities: list[str]) -> float:
        """
        Calculates a significance score based on whether primary tickers or other entities are found.
        Returns a score between 0.0 and 1.0.
        """
        if not entities:
            return 0.0

        entity_set = {ent.upper() for ent in entities}

        # Check for intersection with high-value primary tickers first.
        if self.primary_tickers and not entity_set.isdisjoint(self.primary_tickers):
            return self.params.get('primary_ticker_bonus', 1.0)

        # If no primary tickers are found, grant a smaller bonus for any entity.
        if entity_set:
            return self.params.get('other_entity_bonus', 0.2)

        return 0.0

    def score(self, sentiment_details: dict[str, float], keywords: list[str], entities: list[str]) -> float:
        """
        Computes the final, comprehensive score for a news item.

        Args:
            sentiment_details (Dict[str, float]): The detailed sentiment scores (e.g., {'positive': 0.8, ...}).
            keywords (List[str]): A list of keywords extracted from the text.
            entities (List[str]): A list of named entities (like 'ORG' or 'GPE') from the text.

        Returns:
            float: A final score, where sign is sentiment and magnitude is relevance/significance.
                   Ranges approximately from -1.0 to +1.0.
        """
        # 1. Determine the direction and base magnitude of sentiment (-1 to +1)
        sentiment_direction = self._calculate_directional_sentiment(sentiment_details)

        # 2. Calculate the non-directional magnitude of relevance and significance (0 to 1)
        relevance_magnitude = self._calculate_relevance_score(keywords)
        significance_magnitude = self._calculate_significance_score(entities)

        # 3. Combine magnitudes using configured weights
        w_relevance = self.weights.get('relevance', 0.6)
        w_significance = self.weights.get('significance', 0.4)

        # Weighted average of the two magnitude scores
        total_magnitude = (relevance_magnitude * w_relevance) + (significance_magnitude * w_significance)

        # 4. Amplify the sentiment direction by the calculated magnitude.
        # The formula is: Direction * (1 + Magnitude) to scale the score.
        # This means a highly relevant article (magnitude > 0) will have its sentiment amplified.
        # An article with zero relevance and significance will just have its base sentiment score.
        final_score = sentiment_direction * (1 + total_magnitude)

        # Clamp the result to a standard [-1, 1] range for consistency.
        final_score = max(-1.0, min(final_score, 1.0))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Scoring complete: Direction={sentiment_direction:.2f}, Relevance={relevance_magnitude:.2f}, "
                f"Significance={significance_magnitude:.2f} -> Final Score={final_score:.2f}"
            )
        return round(final_score, 4)
