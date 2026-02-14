# config/source_quality.py

from config.config_loader import load_yaml_config
from typing import Dict, List
import logging



# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def get_rss_feeds() -> Dict[str, str]:
    """
    Load RSS feeds from news_sources.yaml.
    
    Returns:
        Dict mapping feed names to RSS URLs
    """
    try:
        config = load_yaml_config("config/news_sources.yaml")
        return config.get("rss", {})
    except Exception as e:
        logger.error(f"Error loading RSS feeds: {e}")
        return {}

def get_rss_keywords() -> Dict[str, List[str]]:
    """
    Load RSS keywords for filtering from source_quality.yaml.
    
    Returns:
        Dict with keyword categories for RSS filtering
    """
    try:
        config = load_yaml_config("config/source_quality.yaml")
        google_keywords = config.get("google_news_keywords", {})
        
        # Конвертуємо Google News keywords в RSS format
        keyword_dict = {
            "market_terms": google_keywords.get("market_keywords", []),
            "finance_economy": google_keywords.get("economic_keywords", []),
            "sectors_tech": google_keywords.get("tech_keywords", []),
            "professional_indicators": google_keywords.get("stock_keywords", [])
        }
        
        return keyword_dict
        
    except Exception as e:
        logger.error(f"Error loading RSS keywords: {e}")
        return {
            "market_terms": ["etf", "s&p 500", "nasdaq", "dow jones"],
            "finance_economy": ["inflation", "interest rates", "federal reserve", "gdp"],
            "sectors_tech": ["technology", "software", "hardware", "semiconductor"],
            "professional_indicators": ["bullish", "bearish", "buy", "sell"]
        }

def get_google_news_config() -> tuple:
    """
    Get Google News configuration.
    
    Returns:
        Tuple of (keywords_list, source_quality_dict)
    """
    try:
        keywords_config = get_google_news_keywords()
        stock_keywords = keywords_config.get("stock_keywords", [])
        
        source_weights = load_source_quality_config()
        
        return stock_keywords, source_weights
        
    except Exception as e:
        logger.error(f"Error loading Google News config: {e}")
        return [], {}

def load_source_quality_config() -> Dict[str, float]:
    """
    Load source quality weights from source_quality.yaml.
    
    Returns:
        Dict mapping source names to quality weights (0.0 - 1.0)
    """
    try:
        config = load_yaml_config("config/source_quality.yaml")
        source_weights = config.get("source_weights", {})
        
        # Ensure all sources have weights
        default_weights = {
            "unknown": 0.2, "local news": 0.2, "blogs": 0.2, "forums": 0.2,
            "social media": 0.2
        }
        
        # Merge with loaded config, YAML takes precedence over defaults
        merged_weights = {**default_weights, **source_weights}
        
        return merged_weights
        
    except Exception as e:
        # Fallback to basic weights if config loading fails
        return {
            "bloomberg": 1.0, "reuters": 1.0, "wsj": 1.0, "ft": 1.0,
            "cnbc": 1.0, "bbc": 1.0, "marketwatch": 0.6, "business insider": 0.6,
            "yahoo finance": 0.6, "investing.com": 0.6,
            "seeking alpha": 0.4, "benzinga": 0.4,
            "unknown": 0.2
        }


def get_google_news_keywords() -> Dict[str, list]:
    """
    Load Google News keywords from source_quality.yaml.
    
    Returns:
        Dict with keyword categories
    """
    try:
        config = load_yaml_config("config/source_quality.yaml")
        return config.get("google_news_keywords", {})
    except Exception:
        # Fallback keywords
        return {
            "stock_keywords": ["tesla", "tsla", "nvidia", "nvda", "apple", "aapl"],
            "market_keywords": ["stock market", "trading", "earnings", "revenue"],
            "tech_keywords": ["artificial intelligence", "ai", "technology"],
            "economic_keywords": ["inflation", "interest rates", "federal reserve"]
        }


def get_classification_thresholds() -> Dict[str, float]:
    """
    Load classification thresholds from source_quality.yaml.
    
    Returns:
        Dict with threshold values
    """
    try:
        config = load_yaml_config("config/source_quality.yaml")
        return config.get("classification_thresholds", {
            "trash_similarity_threshold": 0.6,
            "importance_threshold": 0.7,
            "neutral_important_threshold": 0.5
        })
    except Exception:
        # Fallback thresholds
        return {
            "trash_similarity_threshold": 0.6,
            "importance_threshold": 0.7,
            "neutral_important_threshold": 0.5
        }


def get_reverse_impact_settings() -> Dict[str, float]:
    """
    Load reverse impact analysis settings from source_quality.yaml.
    
    Returns:
        Dict with reverse impact settings
    """
    try:
        config = load_yaml_config("config/source_quality.yaml")
        return config.get("reverse_impact", {
            "price_threshold": 0.02,
            "sentiment_threshold": 0.1,
            "min_signals_for_analysis": 5,
            "impact_strength_cap": 3.0
        })
    except Exception:
        # Fallback settings
        return {
            "price_threshold": 0.02,
            "sentiment_threshold": 0.1,
            "min_signals_for_analysis": 5,
            "impact_strength_cap": 3.0
        }
