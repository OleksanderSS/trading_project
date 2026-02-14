# enrichment/__init__.py

from .base_sentiment_analyzer import BaseSentimentAnalyzer
from .sentiment_factory import SentimentFactory
from .enrichment_config import EnrichmentConfig

# Convenience imports
from .simple_sentiment import SimpleSentimentAnalyzer

# Optional imports with fallback handling
try:
    from .roberta_sentiment import RobertaSentimentAnalyzer
except ImportError:
    RobertaSentimentAnalyzer = None

from .keyword_extractor import KeywordExtractor
from .entity_extractor import EntityExtractor
from .reverse_impact_analyzer import ReverseImpactAnalyzer
from .summarizer import Summarizer

__all__ = [
    'BaseSentimentAnalyzer',
    'SentimentFactory', 
    'EnrichmentConfig',
    'SimpleSentimentAnalyzer',
    'RobertaSentimentAnalyzer',
    'KeywordExtractor',
    'EntityExtractor',
    'ReverseImpactAnalyzer',
    'Summarizer'
]
