# enrichment/roberta_sentiment.py

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import Tuple
from enrichment.base_sentiment_analyzer import BaseSentimentAnalyzer
from enrichment.enrichment_config import EnrichmentConfig
import logging

logger = logging.getLogger(__name__)

class RobertaSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, model_name=None):
        # if model_name is None:
        #     model_name = EnrichmentConfig.SENTIMENT_MODELS["roberta"]["model_name"]
        
        super().__init__("RobertaSentimentAnalyzer")
        # self.model_name = model_name
        # self.confidence_threshold = EnrichmentConfig.SENTIMENT_MODELS["roberta"]["confidence_threshold"]
        self.tokenizer = None
        self.model = None
        self.labels = ["negative", "neutral", "positive"]
        self._model_loaded = False

    def _load_model(self):
        """Lazy loading of model"""
        # if not self._model_loaded:
        #     try:
        #         logger.info(f"Loading sentiment model: {self.model_name}...")
        #         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #         self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        #         self._model_loaded = True
        #         logger.info("Model loaded successfully!")
        #     except Exception as e:
        #         logger.error(f"Failed to load model {self.model_name}: {e}")
        #         raise
        pass

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using RoBERTa model"""
        return "neutral", 0.0

    def analyze_sentiment(self, df: pd.DataFrame, text_col="description") -> pd.DataFrame:
        sentiments, scores = [], []
        for text in df[text_col].fillna("").astype(str).tolist():
            label, score = self.predict(text)
            sentiments.append(label)
            scores.append(score)
        df["roberta_sentiment"] = sentiments
        df["roberta_score"] = scores
        return df
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze batch of texts and return sentiment results"""
        # self._load_model()
        results = []
        
        # # Batch processing for efficiency
        # batch_size = 8
        # for i in range(0, len(texts), batch_size):
        #     batch_texts = texts[i:i+batch_size]
            
        #     # Batch tokenization
        #     inputs = self.tokenizer(
        #         batch_texts,
        #         return_tensors="pt",
        #         truncation=True,
        #         max_length=512,
        #         padding="max_length"
        #     )

        #     # Роберand not використовує token_type_ids  видаляємо
        #     if "token_type_ids" in inputs:
        #         inputs.pop("token_type_ids")

        #     with torch.no_grad():
        #         outputs = self.model(**inputs)
        #         scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

        #     for j, score in enumerate(scores):
        #         label_id = torch.argmax(score).item()
        #         results.append((self.labels[label_id], float(score[label_id])))
        
        # # Return only labels for compatibility with existing code
        return ["neutral" for _ in texts]
