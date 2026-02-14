# utils/sentiment/finbert_pipeline.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from threading import Lock
from typing import Optional
from utils.logger import ProjectLogger
from utils.logger import ProjectLogger


logger = ProjectLogger.get_logger("TradingProjectLogger")

_FINBERT_PIPELINE: Optional[pipeline] = None
_LOCK = Lock()
_DEVICE = None


def get_finbert_pipeline(device_preference: str = "auto") -> Optional[pipeline]:
    """
    Synchronously returns FinBERT pipeline.
    - Lazy loading, blocks on lock.
    - If failed, raises RuntimeError.
    """
    global _FINBERT_PIPELINE, _DEVICE

    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE

    with _LOCK:
        if _FINBERT_PIPELINE is not None:
            return _FINBERT_PIPELINE

        try:
            if device_preference == "auto":
                _DEVICE = 0 if torch.cuda.is_available() else -1
            elif device_preference == "cpu":
                _DEVICE = -1
            elif device_preference == "cuda":
                _DEVICE = 0
            else:
                raise ValueError(f"Unknown device_preference: {device_preference}")

            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            _FINBERT_PIPELINE = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=_DEVICE
            )
            logger.info(f"[OK] FinBERT loaded ({'cuda' if _DEVICE == 0 else 'cpu'})")
        except Exception as e:
            logger.exception(f"[ERROR] Error loading FinBERT: {e}")
            _FINBERT_PIPELINE = None
            raise RuntimeError("FinBERT pipeline loading failed") from e

    return _FINBERT_PIPELINE
