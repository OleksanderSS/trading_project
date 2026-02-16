# enrichment/keyword_extractor.py

from typing import List, Dict, Union, Optional
import logging
import re
import itertools

logger = logging.getLogger(__name__)

# =========================================================================
# Filters for filtering noise that pretends to be tickers
# =========================================================================

# Common English stop words
COMMON_STOP_WORDS = {
    "THE", "AND", "TO", "OF", "IN", "A", "IS", "IT", "ON", "FOR",
    "THAT", "WITH", "WAS", "ARE", "AS", "BY", "AT", "OR", "FROM",
    "BUT", "BE", "HAVE", "HAD", "THIS", "WILL", "HAS", "NOT", "WE",
    "YOU", "ME", "US", "AM", "AN", "SO", "NO", "UP", "DOWN", "OUT",
    "IF", "MY", "YOUR", "HIS", "HER", "OUR", "THEY", "THEIR"
}

# Financial and general "noise" words that can be in uppercase
FINANCIAL_NOISE_WORDS = {
    "NEWS", "REPORT", "BANK", "GOVT", "TODAY", "WEEK", "MONTH", "YEAR", "FEDERAL",
    "RESERVE", "FOMC", "CNBC", "BBC", "CEO", "CFO", "COO", "USA", "SEC", "IRS",
    "FEDS", "DOJ", "EU", "UK", "USD", "EUR", "JPY", "CHINA", "INDIA", "MARKET",
    "STOCKS", "FUND", "INDEX", "ETF", "GIVE"  # 'GIVE' was in logs
}

# Combine all noise words in uppercase
ALL_NOISE_WORDS = COMMON_STOP_WORDS.union(FINANCIAL_NOISE_WORDS)


# =========================================================================
# KeywordExtractor class
# =========================================================================

class KeywordExtractor:
    def __init__(self, keyword_dict: Optional[Union[List[str], Dict[str, List[str]]]] = None):
        """
        Initializes extractor, filtering tickers and keywords.
        """
        self.keywords: List[str] = []
        self.tickers: List[str] = []
        self.keyword_dict = keyword_dict or {}

        if not isinstance(self.keyword_dict, dict):
            logger.warning("[WARN] Initialized without keywords dictionary. Extraction will be empty.")
            return

        # 1. Process Tickers
        tickers_dict = self.keyword_dict.get("tickers", {})
        if isinstance(tickers_dict, dict):
            # Filter keys (typical tickers)
            raw_tickers = list(tickers_dict.keys())
            self.tickers = [
                t for t in raw_tickers
                if t.isupper() and len(t) >= 2 and t not in ALL_NOISE_WORDS
            ]

            # 2. Process Keywords (groups and aliases)
            flat_values = []
            for group_vals in self.keyword_dict.values():
                if isinstance(group_vals, dict):  # for tickers: dict of lists
                    for vals in group_vals.values():
                        flat_values.extend(vals if isinstance(vals, list) else [vals])
                elif isinstance(group_vals, list):  # for other groups: list
                    flat_values.extend(group_vals)

            # Filter keywords: convert to lowercase,
            # remove empty and noise words
            self.keywords = list(set([
                kw.lower() for kw in flat_values
                if kw and isinstance(kw, str)
                   and kw.upper() not in ALL_NOISE_WORDS
                   and kw.upper() not in self.tickers
            ]))

        # Final logging
        if not self.keywords and not self.tickers:
            logger.warning("[WARN] Initialized without relevant keywords - filtering will be empty.")
        else:
            logger.info(f"[OK] Initialized with {len(self.keywords)} keywords and {len(self.tickers)} tickers (after filtering).")

    def count(self) -> int:
        return len(self.keyword_dict)

    def extract_keywords(self, text: str, **kwargs) -> List[str]:
        """
        Extract keywords and tickers from text using filtered lists.
        """
        if not text or not str(text).strip():
            return []

        # [OK] Tickers (SPY, QQQ) searching in uppercase.
        # Use r'\b' for exact match at word boundary.
        text_upper = text.upper()
        matched_tickers = [
            t for t in self.tickers
            if re.search(rf"\b{re.escape(t)}\b", text_upper)
        ]

        # [OK] Keywords (inflation, ai) searching in lowercase,
        # allowing suffixes (for example, "chip" -> "chips")
        text_lower = text.lower()
        matched_keywords = [
            kw for kw in self.keywords
            if re.search(rf"\b{re.escape(kw)}\w*\b", text_lower)
        ]

        # Merge, remove duplicates and return result
        matched = list(set(matched_tickers + matched_keywords))

        logger.debug(f"[SEARCH] Found {len(matched)} matches in text: {matched}")
        return matched

    def calculate_relevance(self, text: str) -> Dict[str, Union[float, List[str]]]:
        extracted = self.extract_keywords(text)
        return {
            "score": len(extracted),
            "keywords": extracted
        }
