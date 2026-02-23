# utils/mention_utils.py

from enrichment.keyword_extractor import ALL_NOISE_WORDS
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

TICKER_ALIASES = {
    'TSLA': ['tesla', 'tsla', 'elon musk', 'ev maker'],
    'NVDA': ['nvidia', 'nvda', 'jensen huang', 'blackwell', 'h100', 'gpu'],
    'SPY': ['s&p 500', 'spy', 'spx', 'broad market', 'stock market'],
    'QQQ': ['nasdaq', 'qqq', 'tech stocks', 'technology sector']
}


def safe_get(data: Union[Dict, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary or object attribute
    
    Args:
        data: Dictionary or object to extract value from
        key: Key or attribute name to extract
        default: Default value if key not found
        
    Returns:
        Extracted value or default
    """
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        else:
            return getattr(data, key, default)
    except (AttributeError, TypeError) as e:
        logger.debug(f"Failed to get {key}: {e}")
        return default

def compute_mention_score(text: str, tickers: list[str]) -> int:
    """
    Counts the number of ticker mentions in text, excluding stop words and noise tokens.
    """
    if not isinstance(text, str):
        return 0

    tokens = str(text).upper().split()
    # exclude noise words
    tokens = [t for t in tokens if t not in ALL_NOISE_WORDS]

    # count only valid tickers
    return sum(1 for t in tokens if t in [tk.upper() for tk in tickers])

def safe_get_nested(d: dict, *keys, default=None):
    """
    Safe retrieval of nested keys from dictionary.
    """
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current
