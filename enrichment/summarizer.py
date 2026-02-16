# enrichment/summarizer.py

# from transformers import pipeline # Modified by Gemini
from typing import Optional
import logging
# import torch # Modified by Gemini

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", strict: bool = False):
        self.strict = strict
        self.summarizer = None
        try:
            # Modified by Gemini: This will now fail gracefully
            # and the fallback will be used.
            # self.summarizer = pipeline(
            #     "summarization", 
            #     model="t5-small",
            #     device=0 if torch.cuda.is_available() else -1,
            #     framework="pt"
            # )
            # logger.info(f"[OK] Summarizer model 't5-small' successfully loaded")
            raise ImportError("Transformers not available")
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
            return text[:100]  # Returns the first 100 chars

        # FIX: Better logic for short texts
        if len(text.split()) < 15:
            return text
        
        # FIX: Trim text if too long
        if len(text) > 1024:
            text = text[:1024] + "..."

        try:
            # FIX: Use T5 format
            result = self.summarizer(
                f"summarize: {text}",
                max_length=max_length,
                min_length=min_length,
                truncation=True,
                do_sample=False
            )
            
            # FIX: Handle different output formats
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "").strip()
            elif isinstance(result, dict):
                summary = result.get("summary_text", "").strip()
            else:
                summary = str(result).strip()
            
            # If summary is empty, return part of text
            if not summary:
                logger.warning("[WARN] Model returned empty summary, using part of text")
                return text[:max_length]
            
            return summary
            
        except Exception as e:
            logger.warning(f"[ERROR] Error summarize(): {e}")
            return text[:max_length]  # Return part of text as fallback
