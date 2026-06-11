
# src/feature_engineering/nlp/keyword_extractor.py

import logging
import re

logger = logging.getLogger(__name__)

# A more comprehensive set of stop/noise words to avoid false positives
DEFAULT_NOISE_WORDS = {
    "THE", "AND", "TO", "OF", "IN", "A", "IS", "IT", "ON", "FOR", "THAT", "WITH",
    "WAS", "ARE", "AS", "BY", "AT", "OR", "FROM", "BUT", "BE", "HAVE", "HAD", "THIS",
    "WILL", "HAS", "NOT", "WE", "YOU", "ME", "US", "AM", "AN", "SO", "NO", "UP",
    "DOWN", "OUT", "IF", "MY", "YOUR", "HIS", "HER", "OUR", "THEY", "THEIR",
    # Financial & common uppercase words that are not tickers
    "NEWS", "REPORT", "BANK", "GOVT", "TODAY", "WEEK", "MONTH", "YEAR", "FEDERAL",
    "RESERVE", "FOMC", "CNBC", "BBC", "CEO", "CFO", "COO", "USA", "SEC", "IRS",
    "FEDS", "DOJ", "EU", "UK", "USD", "EUR", "JPY", "CHINA", "INDIA", "MARKET",
    "STOCKS", "SHARES", "FUND", "INDEX", "ETF", "GIVE", "CALL", "PUT", "SELL", "BUY"
}

class KeywordExtractor:
    """
    Efficiently extracts pre-defined keywords and tickers from text using compiled regex.
    """

    def __init__(self, keyword_config: dict[str, list[str]] | None = None, noise_words: set[str] | None = None):
        """
        Initializes the extractor and compiles regex for efficient searching.

        Args:
            keyword_config (Optional[Dict[str, List[str]]]): A dictionary where keys are categories
                (e.g., 'tickers', 'technologies') and values are lists of keywords.
            noise_words (Optional[Set[str]]): A set of uppercase words to ignore during extraction.
        """
        self.noise_words = noise_words if noise_words is not None else DEFAULT_NOISE_WORDS
        self.tickers: list[str] = []
        self.keywords: list[str] = []
        self.keyword_regex: re.Pattern | None = None
        self.ticker_regex: re.Pattern | None = None

        if keyword_config:
            self.update_keywords(keyword_config)

    def update_keywords(self, keyword_config: dict[str, list[str]]):
        """
        Updates the keyword lists and recompiles the regex.
        This allows for dynamic updates without creating a new instance.
        """
        if not isinstance(keyword_config, dict):
            logger.warning("Keyword config must be a dictionary. No keywords updated.")
            return

        # Process tickers: typically uppercase, 2-5 chars, not in noise words
        raw_tickers = keyword_config.get('tickers', [])
        self.tickers = sorted({
            t for t in raw_tickers
            if isinstance(t, str) and 2 <= len(t) <= 5 and t.isupper() and t not in self.noise_words
        })

        # Process other keywords: lowercase, not noise, not tickers
        other_keywords = []
        for category, kws in keyword_config.items():
            if category != 'tickers':
                other_keywords.extend(kws)

        ticker_set = set(self.tickers)
        self.keywords = sorted({
            kw.lower() for kw in other_keywords
            if isinstance(kw, str) and kw.upper() not in self.noise_words and kw.upper() not in ticker_set
        })

        # --- Compile Regex for Performance ---
        if self.keywords:
            # Search for keywords case-insensitively, matching whole words
            self.keyword_regex = re.compile(r"\b(" + "|".join(map(re.escape, self.keywords)) + r")\b", re.IGNORECASE)

        if self.tickers:
            # Search for tickers case-sensitively, matching whole words
            self.ticker_regex = re.compile(r"\b(" + "|".join(map(re.escape, self.tickers)) + r")\b")

        logger.info(f"Extractor updated with {len(self.tickers)} tickers and {len(self.keywords)} keywords.")

    def extract(self, text: str) -> list[str]:
        """
        Extracts all configured keywords and tickers from the given text.

        Args:
            text (str): The text to search within.

        Returns:
            List[str]: A sorted list of unique keywords and tickers found in the text.
        """
        # ✅ FIX: Use instance-level dict cache instead of lru_cache (avoids memory leak via self ref)
        if not hasattr(self, '_extract_cache'):
            self._extract_cache: dict = {}
        if text in self._extract_cache:
            return self._extract_cache[text]

        if not text or not isinstance(text, str):
            return []

        found_matches = set()

        # Find tickers using the compiled case-sensitive regex
        if self.ticker_regex:
            found_matches.update(self.ticker_regex.findall(text))

        # Find keywords using the compiled case-insensitive regex
        if self.keyword_regex:
            found_matches.update(match.lower() for match in self.keyword_regex.findall(text))

        result = sorted(found_matches) if found_matches else []

        # Store in instance cache (limit size to avoid unbounded growth)
        if len(self._extract_cache) < 1024:
            self._extract_cache[text] = result

        return result

