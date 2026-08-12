import logging
from typing import List, Dict, Union

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
except ImportError:
    logging.error("Для роботи FinBERT потрібні: torch, transformers")

logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    """
    Аналізатор Сентименту на базі FinBERT.
    Швидка обробка новин або соцмереж (Reddit/Twitter) для отримання числових метрик.
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Завантаження FinBERT ({self.model_name}) на {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.labels = ["positive", "negative", "neutral"]
        except Exception as e:
            logger.error(f"Помилка завантаження FinBERT: {e}")

    def analyze(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        """
        Повертає ймовірності для positive, negative, neutral.
        Якщо на вхід подано один рядок, він буде загорнутий у список.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return []

        # Токенізація з обрізанням до макс. довжини
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Перетворюємо логіти в ймовірності (Softmax)
            probs = F.softmax(outputs.logits, dim=-1)

        results = []
        for prob in probs:
            prob_dict = {
                "positive": float(prob[0]),
                "negative": float(prob[1]),
                "neutral": float(prob[2]),
                # Обчислюємо composite score (від -1 до 1)
                "composite_score": float(prob[0] - prob[1])
            }
            results.append(prob_dict)

        return results

# Синглтон для зручності імпорту в інші скрипти
# analyzer = FinBERTSentimentAnalyzer()
