# enrichment/summarizer.py

from transformers import pipeline
from typing import Optional
import logging
import torch

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", strict: bool = False):
        self.strict = strict
        self.summarizer = None
        try:
            # Тепер torch available for перевandрки пристрою
            self.summarizer = pipeline(
                "summarization", 
                model="t5-small",  # Легша model
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            logger.info(f"[OK] Summarizer model 't5-small' successfully loaded")
        except Exception as e:
            msg = f"[ERROR] Failed to load model {model_name}: {e}"
            if self.strict:
                raise RuntimeError(msg)
            logger.warning(msg)

    def summarize(
        self,
        text: str,
        max_length: int = 100,
        min_length: int = 30
    ) -> Optional[str]:
        if not text or not str(text).strip():
            logger.warning("[WARN] summarize() received empty text")
            return ""

        if self.summarizer is None:
            logger.warning("[WARN] Summarizer unavailable, returning text as is")
            return text[:100]  # Поверandємо першand 100 символandв

        # ВИПРАВКА: Краща логandка for коротких текстandв
        if len(text.split()) < 15:
            return text
        
        # ВИПРАВКА: Обрandforємо текст якщо forнадто довгий
        if len(text) > 1024:
            text = text[:1024] + "..."

        try:
            # ВИПРАВКА: Використовуємо T5 формат
            result = self.summarizer(
                f"summarize: {text}",
                max_length=max_length,
                min_length=min_length,
                truncation=True,
                do_sample=False
            )
            
            # ВИПРАВКА: Обробляємо рandwithнand формати вandдповandдand
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "").strip()
            elif isinstance(result, dict):
                summary = result.get("summary_text", "").strip()
            else:
                summary = str(result).strip()
            
            # Якщо summary пустий, поверandємо частину тексту
            if not summary:
                logger.warning("[WARN] Model returned empty summary, using part of text")
                return text[:max_length]
            
            return summary
            
        except Exception as e:
            logger.warning(f"[ERROR] Error summarize(): {e}")
            return text[:max_length]  # Поверandємо частину тексту як fallback