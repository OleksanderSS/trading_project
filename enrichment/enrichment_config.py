# enrichment/enrichment_config.py

from typing import Dict, List, Optional
import os
from pathlib import Path

class EnrichmentConfig:
    """Centralized configuration for enrichment modules"""
    
    # Sentiment Analysis Models
    SENTIMENT_MODELS = {
        "simple": {
            "name": "SimpleSentimentAnalyzer",
            "description": "Fast keyword-based sentiment analysis",
            "dependencies": [],
            "confidence_threshold": 0.5
        },
        "roberta": {
            "name": "RobertaSentimentAnalyzer", 
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "description": "RoBERTa-based sentiment analysis",
            "dependencies": ["transformers", "torch"],
            "confidence_threshold": 0.7,
            "lazy_loading": True
        },
        "finbert": {
            "name": "FinBERTAnalyzer",
            "model_name": "ProsusAI/finbert",
            "description": "Financial domain-specific sentiment analysis", 
            "dependencies": ["transformers", "torch"],
            "confidence_threshold": 0.8,
            "lazy_loading": True
        }
    }
    
    # Entity Extraction
    ENTITY_EXTRACTION = {
        "spacy_model": "en_core_web_sm",
        "disable_components": ["parser", "lemmatizer", "attribute_ruler"],
        "ticker_patterns": {
            'TSLA': r'\b(tesla|tsla|musk|gigafactory)\b',
            'NVDA': r'\b(nvidia|nvda|jensen|huang|blackwell|h100)\b',
            'SPY': r'\b(s&p 500|spy|spx|market|feds)\b',
            'QQQ': r'\b(nasdaq|qqq|tech stocks)\b'
        }
    }
    
    # Impact Analysis
    IMPACT_ANALYSIS = {
        "price_threshold": 0.02,  # 2% price change
        "sentiment_threshold": 0.1,  # -0.1 to 0.1 for neutral
        "volume_threshold": 1.5,  # 50% above average volume
        "time_window_hours": 24
    }
    
    # Summarization
    SUMMARIZATION = {
        "model_name": "t5-small",
        "max_length": 150,
        "min_length": 30,
        "device": "auto"  # auto-detect GPU/CPU
    }
    
    # Caching
    CACHING = {
        "enabled": True,
        "cache_dir": "data/cache/enrichment",
        "ttl_hours": 24,
        "max_size_mb": 500
    }
    
    # Performance
    PERFORMANCE = {
        "batch_size": 32,
        "max_workers": 4,
        "timeout_seconds": 30
    }
    
    @classmethod
    def get_sentiment_model_config(cls, model_name: str) -> Optional[Dict]:
        """Get configuration for specific sentiment model"""
        return cls.SENTIMENT_MODELS.get(model_name)
    
    @classmethod
    def get_available_sentiment_models(cls) -> List[str]:
        """Get list of available sentiment models"""
        return list(cls.SENTIMENT_MODELS.keys())
    
    @classmethod
    def validate_dependencies(cls, model_name: str) -> bool:
        """Check if required dependencies are available"""
        config = cls.get_sentiment_model_config(model_name)
        if not config:
            return False
        
        dependencies = config.get("dependencies", [])
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                logger.warning(f"Dependency {dep} not available for model {model_name}")
                return False
        
        return True
    
    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get cache directory path"""
        cache_dir = Path(cls.CACHING["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @classmethod
    def get_device_config(cls) -> str:
        """Get device configuration for ML models"""
        device = cls.SUMMARIZATION["device"]
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return device
