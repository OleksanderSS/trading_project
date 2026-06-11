"""
NEWS TICKER DETECTOR
NLP for detecting relevant tickers in news
"""

import logging
from collections import defaultdict

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class NewsTickerDetector:
    """
    NLP detector for identifying tickers in news
    """

    def __init__(self, config: dict = None):
        self.logger = logging.getLogger(__name__)

        # Dictionary of companies and their tickers
        self.company_tickers = {
            # Tech Giants
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'meta': 'META',
            'facebook': 'META',

            # Finance
            'jpmorgan': 'JPM',
            'bank of america': 'BAC',
            'wells fargo': 'WFC',
            'goldman sachs': 'GS',

            # Healthcare
            'johnson & johnson': 'JNJ',
            'pfizer': 'PFE',
            'unitedhealth': 'UNH',

            # Energy
            'exxon': 'XOM',
            'exxonmobil': 'XOM',
            'chevron': 'CVX',

            # Consumer
            'procter & gamble': 'PG',
            'coca-cola': 'KO',
            'walmart': 'WMT',
            'home depot': 'HD',

            # Industrial
            'general electric': 'GE',
            '3m': 'MMM',
            'caterpillar': 'CAT',

            # ETFs
            'spdr': 'SPY',
            'ishares': 'IWM',
            'vanguard': 'VTI',
            'gld': 'GLD',
            'tlts': 'TLT'
        }

        # Direct tickers
        self.direct_tickers = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'WFC', 'GS',
            'JNJ', 'PFE', 'UNH', 'XOM', 'CVX', 'PG', 'KO', 'WMT', 'HD', 'GE', 'MMM',
            'CAT', 'SPY', 'QQQ', 'NVDA', 'TSLA', 'DIA', 'IWM', 'VTI', 'GLD', 'TLT'
        }

        # Keywords for financial news
        self.financial_keywords = {
            'market', 'stock', 'trade', 'trading', 'investment', 'investor',
            'share', 'equity', 'portfolio', 'dividend', 'earnings', 'revenue',
            'profit', 'loss', 'bull', 'bear', 'rally', 'crash', 'volatility'
        }

        # Relevance threshold
        self.relevance_threshold = config.get('news_relevance_threshold', 0.7) if config else 0.7
        self.enable_nlp_detection = config.get('enable_nlp_ticker_detection', True) if config else True

    def extract_tickers_from_text(self, text: str) -> list[tuple[str, float]]:
        """
        Extract tickers from news text

        Args:
            text: News text

        Returns:
            List[Tuple[str, float]]: List of (ticker, confidence_score)
        """
        if not self.enable_nlp_detection:
            return []

        text_lower = text.lower()
        text_upper = text.upper()
        found_tickers = []

        # 1. Direct search for tickers
        for ticker in self.direct_tickers:
            if ticker in text_upper:
                # Count the number of mentions
                count = text_upper.count(ticker)
                confidence = min(0.9, 0.5 + count * 0.1)  # More mentions = higher confidence
                found_tickers.append((ticker, confidence))

        # 2. Search for company names
        for company, ticker in self.company_tickers.items():
            if company in text_lower:
                # Check if it's not part of another word
                words = text_lower.split()
                for word in words:
                    if company in word and len(word) <= len(company) + 3:
                        confidence = 0.7  # Medium confidence for company names
                        found_tickers.append((ticker, confidence))
                        break

        # 3. Financial relevance
        financial_score = self._calculate_financial_relevance(text_lower)

        # 4. Sort by confidence
        found_tickers.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold
        relevant_tickers = [
            (ticker, min(confidence, financial_score))
            for ticker, confidence in found_tickers
            if confidence >= self.relevance_threshold
        ]

        return relevant_tickers

    def _calculate_financial_relevance(self, text: str) -> float:
        """
        Calculate the financial relevance of the text

        Args:
            text: News text

        Returns:
            float: Score from 0.0 to 1.0
        """
        words = text.split()
        financial_words = sum(1 for word in words if word in self.financial_keywords)

        if len(words) == 0:
            return 0.0

        relevance = financial_words / len(words)
        return min(1.0, relevance * 5)  # Scaling

    def get_primary_ticker(self, text: str, fallback_symbol: str = 'SPY') -> str | None:
        """
        Get the primary ticker for the news

        Args:
            text: News text
            fallback_symbol: Fallback symbol

        Returns:
            Optional[str]: Primary ticker or None if not financial news
        """
        tickers = self.extract_tickers_from_text(text)

        if tickers:
            # Return the ticker with the highest confidence
            return tickers[0][0]

        # If no tickers are found, check financial relevance
        financial_score = self._calculate_financial_relevance(text.lower())

        if financial_score >= self.relevance_threshold:
            # Financially relevant news without a specific ticker
            return fallback_symbol

        # Not financial news
        return None

    def analyze_news_batch(self, news_data: list[dict]) -> list[dict]:
        """
        Analyze a batch of news and add tickers

        Args:
            news_data: List of news

        Returns:
            List[Dict]: News with added tickers
        """
        analyzed_news = []

        for news in news_data:
            text = news.get('title', '') + ' ' + news.get('content', '')

            # Extract tickers
            tickers = self.extract_tickers_from_text(text)
            primary_ticker = self.get_primary_ticker(text)

            # Update the news item
            enhanced_news = news.copy()
            enhanced_news['detected_tickers'] = [ticker for ticker, _ in tickers]
            enhanced_news['ticker_confidence'] = dict(tickers)
            enhanced_news['primary_ticker'] = primary_ticker
            enhanced_news['financial_relevance'] = self._calculate_financial_relevance(text.lower())

            analyzed_news.append(enhanced_news)

        return analyzed_news

    def get_ticker_distribution(self, news_data: list[dict]) -> dict[str, int]:
        """
        Get the distribution of tickers in the news

        Args:
            news_data: List of news

        Returns:
            Dict[str, int]: Number of news items per ticker
        """
        ticker_counts = defaultdict(int)

        for news in news_data:
            primary_ticker = news.get('primary_ticker')
            if primary_ticker:
                ticker_counts[primary_ticker] += 1

        return dict(ticker_counts)

    def filter_relevant_news(self, news_data: list[dict],
                           min_relevance: float = None) -> list[dict]:
        """
        Filter relevant news

        Args:
            news_data: List of news
            min_relevance: Minimum relevance

        Returns:
            List[Dict]: Relevant news
        """
        if min_relevance is None:
            min_relevance = self.relevance_threshold

        relevant_news = []

        for news in news_data:
            financial_relevance = news.get('financial_relevance', 0.0)
            has_ticker = news.get('primary_ticker') is not None

            if financial_relevance >= min_relevance or has_ticker:
                relevant_news.append(news)

        return relevant_news

def create_news_ticker_detector(config: dict = None) -> NewsTickerDetector:
    """
    Factory function for creating the detector

    Args:
        config: Configuration

    Returns:
        NewsTickerDetector: Detector instance
    """
    return NewsTickerDetector(config)
