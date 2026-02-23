# enrichment/base_sentiment_analyzer.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseSentimentAnalyzer(ABC):
    """Base class for all sentiment analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self._model_loaded = False
        
    @abstractmethod
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for given text
        
        Returns:
            Tuple[str, float]: (sentiment_label, confidence_score)
        """
        pass
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model (lazy loading)"""
        pass
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting sentiment for text: {e}")
                results.append(("neutral", 0.0))
        return results
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "name": self.name,
            "loaded": self._model_loaded,
            "type": self.__class__.__name__
        }
