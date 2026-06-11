# src/sentiment/sentiment_models.py

import hashlib
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_FINBERT_PIPELINE = None
_TOKENIZER = None
_CACHE: dict[str, dict[str, str]] = {}  # result cache by text hash

def _stable_hash(text: str) -> str:
    """Short hash of text for caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def get_finbert_pipeline(device: int | None = None) -> Any:
    """Returns the global FinBERT pipeline. Loads once."""
    global _FINBERT_PIPELINE, _TOKENIZER
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE

    try:
        # --- DYNAMIC IMPORTS ---
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        # ----------------------

        # ✅ HuggingFace timeout settings
        # Increasing timeout for model download
        os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')  # 5 minutes

        # Use token if available
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            logger.info("✅ Using HF_TOKEN for authentication")

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        logger.info("📥 Downloading FinBERT tokenizer...")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert",
            token=hf_token,
        )

        logger.info("📥 Downloading FinBERT model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            token=hf_token,
        )

        logger.info("🔧 Creating pipeline...")
        _FINBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=_TOKENIZER,
            device=device
        )
        logger.info(f"✅ FinBERT loaded successfully ({'cuda' if device == 0 else 'cpu'})")
    except ImportError:
        logger.warning("⚠️ Libraries 'torch' or 'transformers' not found. Sentiment analysis will be disabled.")
        _FINBERT_PIPELINE = "disabled" # Mark as checked and disabled
    except Exception as e:
        logger.error(f"❌ Error loading FinBERT: {e}", exc_info=True)
        logger.warning("⚠️ FinBERT disabled. Sentiment analysis will return neutral for all texts.")
        _FINBERT_PIPELINE = "disabled"

    return _FINBERT_PIPELINE

def analyze_sentiment(texts: list[str], batch_size: int = 16, device: int | None = None, **kwargs: Any) -> pd.DataFrame:
    """
    Analyzes sentiment of a list of texts in batches.
    Uses caching for repeated texts.
    Returns DataFrame: text, label, score.
    """
    pipe = get_finbert_pipeline(device=device)
    if pipe is None or pipe == "disabled":
        return _create_neutral_dataframe(texts)

    rows: list[dict[str, Any]] = []
    label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_safe = _prepare_batch_texts(batch)

        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # 1. Fill cached rows, identify uncached
        for idx, t in enumerate(batch):
            h = _stable_hash(t)
            if h in _CACHE:
                rows.append(_CACHE[h])
            else:
                uncached_texts.append(batch_safe[idx])
                uncached_indices.append(idx)

        # 2. Process uncached texts
        if uncached_texts:
            try:
                results = pipe(uncached_texts, truncation=True, max_length=512, **kwargs)
                for idx, res in zip(uncached_indices, results, strict=False):
                    row = _create_sentiment_row(res, batch[idx], label_map)
                    rows.append(row)
                    _CACHE[_stable_hash(batch[idx])] = row
            except Exception as e:
                logger.error(f"[WARN] Batch {i} error: {e}", exc_info=True)
                for idx in uncached_indices:
                    row = {"text": batch[idx], "label": "neutral", "score": 0.0}
                    rows.append(row)

    return pd.DataFrame(rows)
def _create_neutral_dataframe(texts: list[str]) -> pd.DataFrame:
    """Create neutral sentiment DataFrame when FinBERT is unavailable."""
    return pd.DataFrame([{"text": t, "label": "neutral", "score": 0.0} for t in texts])

def _prepare_batch_texts(batch: list[str]) -> list[str]:
    """Prepare batch texts by replacing empty strings with 'neutral'."""
    return [t if t.strip() else "neutral" for t in batch]

def _create_sentiment_row(result: dict, text: str, label_map: dict[str, str]) -> dict:
    """Create a sentiment row from pipeline result."""
    label_raw = result.get("label", "error").lower()
    label = label_map.get(label_raw, "neutral")
    score = float(result.get("score", 0.0))
    return {"text": text, "label": label, "score": score}

def aggregate_sentiment(df: pd.DataFrame, normalize: bool = True, method: str = "mean") -> dict[str, float]:
    """
    Computes aggregated sentiment across all news items.
    method: "mean" | "sum" | "count"
    If normalize=True, sum of three categories = 1.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    if df.empty:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    if method == "sum":
        agg = df.groupby("label")["score"].sum().to_dict()
    elif method == "count":
        agg = df["label"].value_counts(normalize=False).to_dict()
    else:  # mean (calculates population-weighted mean to prevent group-size distortions)
        total_count = len(df)
        agg = (df.groupby("label")["score"].sum() / total_count).to_dict()

    res = {
        "positive": float(agg.get("positive", 0.0)),
        "negative": float(agg.get("negative", 0.0)),
        "neutral": float(agg.get("neutral", 0.0)),
    }

    total = sum(res.values())
    if normalize and total > 0:
        res = {k: v / total for k, v in res.items()}
    elif normalize and total == 0:
        logger.warning("[WARN] All sentiment scores = 0, returning 0 for all categories")

    return res
