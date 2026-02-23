# enrichment/sentiment_factory.py

from typing import Dict, List, Optional, Type
from enrichment.base_sentiment_analyzer import BaseSentimentAnalyzer
from enrichment.simple_sentiment import SimpleSentimentAnalyzer
from enrichment.enrichment_config import EnrichmentConfig
import logging

logger = logging.getLogger(__name__)

class SentimentFactory:
    """Factory for creating sentiment analyzers with fallback support"""
    
    _analyzers: Dict[str, Type[BaseSentimentAnalyzer]] = {}
    
    @classmethod
    def register_analyzer(cls, name: str, analyzer_class: Type[BaseSentimentAnalyzer]):
        """Register a sentiment analyzer"""
        cls._analyzers[name] = analyzer_class
    
    @classmethod
    def create_analyzer(cls, preferred_model: str = None) -> BaseSentimentAnalyzer:
        """Create sentiment analyzer with automatic fallback"""
        
        # If no preference, use simple analyzer
        if preferred_model is None:
            preferred_model = "simple"
        
        # Try to create preferred analyzer
        if preferred_model in cls._analyzers:
            if cls._validate_dependencies(preferred_model):
                try:
                    return cls._analyzers[preferred_model]()
                except Exception as e:
                    logger.error(f"Failed to create {preferred_model} analyzer: {e}")
        
        # Fallback to simple analyzer
        logger.warning(f"Falling back to simple sentiment analyzer")
        return SimpleSentimentAnalyzer()
    
    @classmethod
    def _validate_dependencies(cls, model_name: str) -> bool:
        """Validate if dependencies are available"""
        return EnrichmentConfig.validate_dependencies(model_name)
    
    @classmethod
    def get_analyzer(cls, name: str) -> Optional['BaseSentimentAnalyzer']:
        """Get specific analyzer by name"""
        if name in cls._analyzers:
            try:
                return cls._analyzers[name]()
            except Exception as e:
                logger.error(f"Failed to create {name} analyzer: {e}")
                return None
        return None
    
    @classmethod
    def get_available_analyzers(cls) -> List[str]:
        """Get list of available analyzers"""
        available = []
        for name in cls._analyzers.keys():
            if cls._validate_dependencies(name):
                available.append(name)
        return available
    
    @classmethod
    def get_analyzer_info(cls) -> Dict:
        """Get information about all analyzers"""
        info = {}
        for name, analyzer_class in cls._analyzers.items():
            try:
                analyzer = analyzer_class()
                info[name] = analyzer.get_model_info()
            except Exception as e:
                info[name] = {"error": str(e)}
        return info

# Register analyzers
SentimentFactory.register_analyzer("simple", SimpleSentimentAnalyzer)

# Try to register advanced analyzers with optional imports
try:
    from enrichment.roberta_sentiment import RobertaSentimentAnalyzer
    SentimentFactory.register_analyzer("roberta", RobertaSentimentAnalyzer)
except ImportError:
    logger.warning("Roberta analyzer not available (missing dependencies)")

try:
    # Add FinBERT analyzer
    from enrichment.finbert_analyzer import FinBERTAnalyzer
    SentimentFactory.register_analyzer("finbert", FinBERTAnalyzer)
    logger.info("[FinBERT] Analyzer registered successfully")
    
except ImportError as e:
    logger.warning(f"FinBERT analyzer not available (missing dependencies): {e}")
except Exception as e:
    logger.error(f"FinBERT analyzer registration failed: {e}")
