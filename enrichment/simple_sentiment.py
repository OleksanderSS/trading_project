# enrichment/simple_sentiment.py

import pandas as pd
import re
from typing import List, Tuple
from enrichment.base_sentiment_analyzer import BaseSentimentAnalyzer
from enrichment.enrichment_config import EnrichmentConfig

class SimpleSentimentAnalyzer(BaseSentimentAnalyzer):
    """Lightweight sentiment analyzer using keyword-based approach"""
    
    def __init__(self):
        super().__init__("SimpleSentimentAnalyzer")
        self.confidence_threshold = EnrichmentConfig.SENTIMENT_MODELS["simple"]["confidence_threshold"]
        
        # Positive keywords
        self.positive_words = {
            'good', 'great', 'excellent', 'positive', 'bullish', 'buy', 'strong', 
            'growth', 'profit', 'gain', 'up', 'rise', 'increase', 'win', 'success',
            'breakthrough', 'innovation', 'record', 'high', 'best', 'top', 'leader'
        }
        
        # Negative keywords
        self.negative_words = {
            'bad', 'terrible', 'negative', 'bearish', 'sell', 'weak', 'loss', 
            'decline', 'fall', 'decrease', 'drop', 'crash', 'risk', 'threat',
            'concern', 'worry', 'fear', 'panic', 'crisis', 'recession', 'low',
            'worst', 'fail', 'struggle', 'problem', 'issue', 'challenge'
        }
        
        self.labels = ["negative", "neutral", "positive"]
        self._model_loaded = True  # No model to load
    
    def _load_model(self):
        """No model to load for simple analyzer"""
        pass
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using keyword analysis"""
        if not text or not isinstance(text, str):
            return "neutral", 0.0

        cleaned_text = self._clean_text(text)
        words = set(cleaned_text.lower().split())
        
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(positive_count / max(positive_count + negative_count, 1), 1.0)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(negative_count / max(positive_count + negative_count, 1), 1.0)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return sentiment, confidence
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for single text"""
        if not text or not isinstance(text, str):
            return "neutral", 0.0
        
        cleaned_text = self._clean_text(text)
        words = cleaned_text.split()
        
        if not words:
            return "neutral", 0.0
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return sentiment, confidence
    
    def analyze_batch(self, texts: List[str]) -> List[str]:
        """Analyze batch of texts and return sentiment labels"""
        results = []
        for text in texts:
            sentiment, _ = self.predict(text)
            results.append(sentiment)
        return results
    
    def analyze_sentiment(self, df: pd.DataFrame, text_col="description") -> pd.DataFrame:
        """Analyze sentiment for DataFrame"""
        sentiments, scores = [], []
        for text in df[text_col].fillna("").astype(str).tolist():
            sentiment, score = self.predict(text)
            sentiments.append(sentiment)
            scores.append(score)
        
        df["simple_sentiment"] = sentiments
        df["simple_score"] = scores
        return df

# Fallback function for compatibility
def create_sentiment_analyzer(use_simple: bool = False):
    """Create sentiment analyzer based on preference"""
    if use_simple:
        return SimpleSentimentAnalyzer()
    else:
        try:
            from enrichment.roberta_sentiment import RobertaSentimentAnalyzer
            return RobertaSentimentAnalyzer()
        except Exception as e:
            print(f"Failed to load Roberta model: {e}")
            print("Using simple sentiment analyzer instead")
            return SimpleSentimentAnalyzer()
