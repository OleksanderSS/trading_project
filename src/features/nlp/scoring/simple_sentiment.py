
# src/features/scoring/simple_sentiment.py

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Default sentiment keywords if none are provided
DEFAULT_POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'positive', 'bullish', 'buy', 'strong', 'growth', 'profit',
    'gains', 'gained', 'up', 'rise', 'increase', 'surge', 'record', 'high', 'beat', 'outperform'
}

DEFAULT_NEGATIVE_WORDS = {
    'bad', 'terrible', 'poor', 'negative', 'bearish', 'sell', 'weak', 'loss', 'decline', 'down',
    'fall', 'decrease', 'drop', 'slump', 'low', 'miss', 'underperform', 'risk', 'warning'
}

class SimpleSentimentAnalyzer:
    """
    A fast, keyword-based sentiment analyzer.
    It determines sentiment by counting positive and negative words in a text.
    """

    def __init__(self, sentiment_config: dict[str, Any] | None = None):
        """
        Initializes the SimpleSentimentAnalyzer.

        Args:
            sentiment_config (Optional[Dict[str, Any]]): A configuration dictionary containing:
                - 'positive_words': A list of words to be considered positive.
                - 'negative_words': A list of words to be considered negative.
        """
        config = sentiment_config or {}

        # Use provided keywords or fall back to defaults safely without triggering set(None) TypeError
        pos_list = config.get('positive_words')
        self.positive_words: set[str] = set(pos_list) if pos_list is not None else DEFAULT_POSITIVE_WORDS

        neg_list = config.get('negative_words')
        self.negative_words: set[str] = set(neg_list) if neg_list is not None else DEFAULT_NEGATIVE_WORDS

        # Regex to find all matching keywords in one pass
        all_words = self.positive_words.union(self.negative_words)
        self.keyword_regex = re.compile(r"\b(" + "|".join(map(re.escape, all_words)) + r")\b", re.IGNORECASE)

        logger.info(
            f"SimpleSentimentAnalyzer initialized with "
            f"{len(self.positive_words)} positive and {len(self.negative_words)} negative words."
        )

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyzes the sentiment of a text based on keyword counts.

        Args:
            text (str): The input text.

        Returns:
            Dict[str, Any]: A dictionary with the sentiment label, score, and details.
        """
        if not text or not isinstance(text, str):
            return {'label': 'neutral', 'score': 1.0, 'details': {'positive': 0, 'negative': 0, 'neutral': 1}}

        # Find all sentimental words in the text
        found_words = self.keyword_regex.findall(text.lower())

        if not found_words:
            return {'label': 'neutral', 'score': 1.0, 'details': {'positive': 0, 'negative': 0, 'neutral': 1}}

        # Count positive and negative occurrences
        pos_count = sum(1 for word in found_words if word in self.positive_words)
        neg_count = sum(1 for word in found_words if word in self.negative_words)

        total_sentiment_words = pos_count + neg_count

        if total_sentiment_words == 0:
             return {'label': 'neutral', 'score': 1.0, 'details': {'positive': 0, 'negative': 0, 'neutral': 1}}

        # Calculate sentiment scores
        pos_score = pos_count / total_sentiment_words
        neg_score = neg_count / total_sentiment_words

        # Determine the final label and score
        if pos_score > neg_score:
            label = 'positive'
            score = pos_score
        elif neg_score > pos_score:
            label = 'negative'
            score = neg_score
        else:
            label = 'neutral'
            score = 1.0  # A neutral score is certain

        return {
            'label': label,
            'score': round(score, 4),
            'details': {
                'positive': round(pos_score, 4),
                'negative': round(neg_score, 4),
                'neutral': round(1.0 - pos_score - neg_score, 4)
            }
        }
