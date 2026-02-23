# enrichment/finbert_analyzer.py

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FinBERTAnalyzer:
    """FinBERT sentiment analyzer for financial text"""
    
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize FinBERT pipeline"""
        try:
            # Import here to avoid circular imports
            from utils.sentiment.finbert_pipeline import get_finbert_pipeline
            self.pipeline = get_finbert_pipeline()
            logger.info("[FinBERT] Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"[FinBERT] Failed to initialize pipeline: {e}")
            self.pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        if not self.pipeline:
            return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0}
        
        try:
            result = self.pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # Convert to standard format
            if label == 'POSITIVE':
                sentiment = 1.0
            elif label == 'NEGATIVE':
                sentiment = -1.0
            else:
                sentiment = 0.0
            
            return {
                "label": label,
                "score": sentiment,
                "confidence": score,
                "raw_score": score
            }
        except Exception as e:
            logger.error(f"[FinBERT] Analysis failed: {e}")
            return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "FinBERTAnalyzer",
            "model_name": "ProsusAI/finbert",
            "description": "Financial domain-specific sentiment analysis",
            "confidence_threshold": self.confidence_threshold,
            "is_initialized": self.pipeline is not None
        }
