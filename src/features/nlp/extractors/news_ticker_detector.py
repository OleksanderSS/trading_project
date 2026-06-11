"""
NEWS TICKER DETECTOR
NLP for detecting relevant tickers in news
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional, TypedDict

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────────────────────

class NewsItem(TypedDict, total=False):
    title: str
    content: str
    primary_ticker: Optional[str]
    financial_relevance: float
    detected_tickers: list[str]
    ticker_confidence: dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class NewsTickerDetector:
    """
    NLP detector for identifying tickers in news.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
    ) -> None:

        self.logger = logging.getLogger(__name__)

        config = config or {}

        # ─────────────────────────────────────────────────────────────────────
        # COMPANY → TICKER MAP
        # ─────────────────────────────────────────────────────────────────────

        self.company_tickers: dict[str, str] = {

            # Tech
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",

            # Finance
            "jpmorgan": "JPM",
            "bank of america": "BAC",
            "wells fargo": "WFC",
            "goldman sachs": "GS",

            # Healthcare
            "johnson & johnson": "JNJ",
            "pfizer": "PFE",
            "unitedhealth": "UNH",

            # Energy
            "exxon": "XOM",
            "exxonmobil": "XOM",
            "chevron": "CVX",

            # Consumer
            "procter & gamble": "PG",
            "coca-cola": "KO",
            "walmart": "WMT",
            "home depot": "HD",

            # Industrial
            "general electric": "GE",
            "3m": "MMM",
            "caterpillar": "CAT",

            # ETFs
            "spdr": "SPY",
            "ishares": "IWM",
            "vanguard": "VTI",
            "gld": "GLD",
            "tlts": "TLT",
        }

        # ─────────────────────────────────────────────────────────────────────
        # DIRECT TICKERS
        # ─────────────────────────────────────────────────────────────────────

        self.direct_tickers: set[str] = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "JNJ",
            "PFE",
            "UNH",
            "XOM",
            "CVX",
            "PG",
            "KO",
            "WMT",
            "HD",
            "GE",
            "MMM",
            "CAT",
            "SPY",
            "QQQ",
            "NVDA",
            "TSLA",
            "DIA",
            "IWM",
            "VTI",
            "GLD",
            "TLT",
        }

        # ─────────────────────────────────────────────────────────────────────
        # FINANCIAL KEYWORDS
        # ─────────────────────────────────────────────────────────────────────

        self.financial_keywords: set[str] = {
            "market",
            "stock",
            "trade",
            "trading",
            "investment",
            "investor",
            "share",
            "equity",
            "portfolio",
            "dividend",
            "earnings",
            "revenue",
            "profit",
            "loss",
            "bull",
            "bear",
            "rally",
            "crash",
            "volatility",
        }

        # ─────────────────────────────────────────────────────────────────────
        # GENERAL ECONOMIC KEYWORDS
        # ─────────────────────────────────────────────────────────────────────

        self.general_keywords: set[str] = {
            "inflation",
            "interest rates",
            "federal reserve",
            "fed",
            "gdp",
            "unemployment",
            "recession",
            "stimulus",
            "bond yields",
            "economy",
            "economic",
            "financial",
            "s&p 500",
            "nasdaq",
            "index",
            "etf",
            "earnings report",
            "quarterly results",
            "financial results",
            "merger",
            "m&a",
            "acquisition",
            "regulation",
            "downgrade",
            "upgrade",
            "buyback",
            "guidance",
            "forecast",
            "ipo",
            "bankruptcy",
            "restructuring",
            "oil",
            "energy",
            "commodities",
            "opec",
            "gas",
            "renewables",
            "sanctions",
            "tariffs",
            "trade war",
            "conflict",
            "china",
            "eu",
            "us fed",
            "antitrust",
            "ai",
            "ev",
            "semiconductor",
            "chips",
            "cloud",
            "tech",
            "cybersecurity",
            "data breach",
            "robotics",
            "automation",
            "battery",
            "charging",
            "fda approval",
            "clinical trial",
            "biotech",
            "pharmaceutical",
            "medical device",
            "banking",
            "insurance",
            "fintech",
            "payments",
            "lending",
        }

        # ─────────────────────────────────────────────────────────────────────
        # CONFIG
        # ─────────────────────────────────────────────────────────────────────

        self.relevance_threshold: float = config.get(
            "news_relevance_threshold",
            0.7,
        )

        self.enable_nlp_detection: bool = config.get(
            "enable_nlp_ticker_detection",
            True,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def extract_tickers_from_text(
        self,
        text: str,
    ) -> list[tuple[str, float]]:

        """
        Extract tickers from news text.
        """

        if not self.enable_nlp_detection:
            return []

        text_lower = text.lower()
        text_upper = text.upper()

        found_tickers: list[tuple[str, float]] = []

        # ─────────────────────────────────────────────────────────────────────
        # DIRECT TICKERS
        # ─────────────────────────────────────────────────────────────────────

        for ticker in self.direct_tickers:

            if ticker in text_upper:

                count = text_upper.count(ticker)

                confidence = min(
                    0.9,
                    0.5 + count * 0.1,
                )

                found_tickers.append(
                    (ticker, confidence)
                )

        # ─────────────────────────────────────────────────────────────────────
        # COMPANY NAMES
        # ─────────────────────────────────────────────────────────────────────

        words = text_lower.split()

        for company, ticker in self.company_tickers.items():

            if company not in text_lower:
                continue

            for word in words:

                if (
                    company in word
                    and len(word) <= len(company) + 3
                ):

                    found_tickers.append(
                        (ticker, 0.7)
                    )

                    break

        # ─────────────────────────────────────────────────────────────────────
        # RELEVANCE
        # ─────────────────────────────────────────────────────────────────────

        general_score = self._calculate_general_relevance(
            text_lower
        )

        financial_score = self._calculate_financial_relevance(
            text_lower
        )

        # ─────────────────────────────────────────────────────────────────────
        # GENERAL NEWS
        # ─────────────────────────────────────────────────────────────────────

        if (
            not found_tickers
            and general_score >= self.relevance_threshold
        ):
            return [("general", general_score)]

        # ─────────────────────────────────────────────────────────────────────
        # SORT + FILTER
        # ─────────────────────────────────────────────────────────────────────

        found_tickers.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        relevant_tickers = [
            (
                ticker,
                min(confidence, financial_score),
            )
            for ticker, confidence in found_tickers
            if confidence >= self.relevance_threshold
        ]

        return relevant_tickers

    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_general_relevance(
        self,
        text: str,
    ) -> float:

        words = text.split()

        if not words:
            return 0.0

        general_words = sum(
            1
            for word in words
            if word in self.general_keywords
        )

        relevance = general_words / len(words)

        return min(1.0, relevance * 3)

    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_financial_relevance(
        self,
        text: str,
    ) -> float:

        words = text.split()

        if not words:
            return 0.0

        financial_words = sum(
            1
            for word in words
            if word in self.financial_keywords
        )

        return min(
            1.0,
            financial_words / len(words) * 2,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def get_primary_ticker(
        self,
        text: str,
        fallback_symbol: str = "SPY",
    ) -> Optional[str]:

        tickers = self.extract_tickers_from_text(text)

        if tickers:
            return tickers[0][0]

        financial_score = self._calculate_financial_relevance(
            text.lower()
        )

        if financial_score >= self.relevance_threshold:
            return fallback_symbol

        return None

    # ─────────────────────────────────────────────────────────────────────────

    def analyze_news_batch(
        self,
        news_data: list[NewsItem],
    ) -> list[NewsItem]:

        analyzed_news: list[NewsItem] = []

        for news in news_data:

            text = (
                news.get("title", "")
                + " "
                + news.get("content", "")
            )

            tickers = self.extract_tickers_from_text(
                text
            )

            primary_ticker = self.get_primary_ticker(
                text
            )

            enhanced_news = news.copy()

            enhanced_news["detected_tickers"] = [
                ticker
                for ticker, _ in tickers
            ]

            enhanced_news["ticker_confidence"] = dict(
                tickers
            )

            enhanced_news["primary_ticker"] = (
                primary_ticker
            )

            enhanced_news["financial_relevance"] = (
                self._calculate_financial_relevance(
                    text.lower()
                )
            )

            analyzed_news.append(enhanced_news)

        return analyzed_news

    # ─────────────────────────────────────────────────────────────────────────

    def get_ticker_distribution(
        self,
        news_data: list[NewsItem],
    ) -> dict[str, int]:

        ticker_counts: defaultdict[str, int] = defaultdict(
            int
        )

        for news in news_data:

            primary_ticker = news.get(
                "primary_ticker"
            )

            if primary_ticker:
                ticker_counts[primary_ticker] += 1

        return dict(ticker_counts)

    # ─────────────────────────────────────────────────────────────────────────

    def filter_relevant_news(
        self,
        news_data: list[NewsItem],
        min_relevance: Optional[float] = None,
    ) -> list[NewsItem]:

        if min_relevance is None:
            min_relevance = self.relevance_threshold

        relevant_news: list[NewsItem] = []

        for news in news_data:

            financial_relevance = news.get(
                "financial_relevance",
                0.0,
            )

            has_ticker = (
                news.get("primary_ticker")
                is not None
            )

            if (
                financial_relevance >= min_relevance
                or has_ticker
            ):
                relevant_news.append(news)

        return relevant_news


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_news_ticker_detector(
    config: Optional[dict[str, Any]] = None,
) -> NewsTickerDetector:

    return NewsTickerDetector(config)