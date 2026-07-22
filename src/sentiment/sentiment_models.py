# src/sentiment/sentiment_models.py

import hashlib

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_FINBERT_PIPELINE = None
_TOKENIZER = None
_CACHE: dict[str, dict[str, str]] = {}  # result cache by text hash

def _stable_hash(text: str) -> str:
    """Short hash of text for caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def get_finbert_pipeline(device: int = None):
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
            timeout=300  # 5 minutes
        )

        logger.info("📥 Downloading FinBERT model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            token=hf_token
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
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"❌ Error loading FinBERT: {e}")
        logger.warning("⚠️ FinBERT disabled. Sentiment analysis will return neutral for all texts.")
        _FINBERT_PIPELINE = "disabled"

    return _FINBERT_PIPELINE

def analyze_sentiment(texts: list[str], batch_size: int = 16, device: int = None, **kwargs) -> pd.DataFrame:
    """
    Analyzes sentiment of a list of texts in batches.
    Uses caching for repeated texts.
    Returns DataFrame: text, label, score.
    """
    pipe = get_finbert_pipeline(device=device)
    if pipe is None or pipe == "disabled":
        return _create_neutral_dataframe(texts)

    rows = []
    label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_safe = _prepare_batch_texts(batch)

        # Check cache and process uncached texts
        uncached_texts, uncached_indices, cached_rows = _check_cache(batch, batch_safe)
        rows.extend(cached_rows)

        if not uncached_texts:
            continue

        # Process uncached texts
        batch_rows = _process_batch(pipe, uncached_texts, uncached_indices, batch, label_map, i, **kwargs)
        rows.extend(batch_rows)

    return pd.DataFrame(rows)

def _create_neutral_dataframe(texts: list[str]) -> pd.DataFrame:
    """Create neutral sentiment DataFrame when FinBERT is unavailable."""
    return pd.DataFrame([{"text": t, "label": "neutral", "score": 0.0} for t in texts])

def _prepare_batch_texts(batch: list[str]) -> list[str]:
    """Prepare batch texts by replacing empty strings with 'neutral'."""
    return [t if t.strip() else "neutral" for t in batch]

def _check_cache(batch: list[str], batch_safe: list[str]) -> tuple[list[str], list[int], list[dict]]:
    """Check cache for processed texts and return uncached items."""
    uncached_texts, uncached_indices, cached_rows = [], [], []

    for idx, t in enumerate(batch):
        h = _stable_hash(t)
        if h in _CACHE:
            cached_rows.append(_CACHE[h])
        else:
            uncached_texts.append(batch_safe[idx])
            uncached_indices.append(idx)

    return uncached_texts, uncached_indices, cached_rows

def _process_batch(pipe, uncached_texts: list[str], uncached_indices: list[int],
                  original_batch: list[str], label_map: dict[str, str], batch_idx: int, **kwargs) -> list[dict]:
    """Process a batch of uncached texts through the sentiment pipeline."""
    rows = []

    try:
        results = pipe(uncached_texts, truncation=True, max_length=512, **kwargs)
        for idx, res in zip(uncached_indices, results, strict=False):
            row = _create_sentiment_row(res, original_batch[idx], label_map)
            rows.append(row)
            _CACHE[_stable_hash(original_batch[idx])] = row
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"[WARN] Batch {batch_idx} error: {e}")
        for idx in uncached_indices:
            rows.append({"text": original_batch[idx], "label": "error", "score": 0.0})

    return rows

def _create_sentiment_row(result: dict, text: str, label_map: dict[str, str]) -> dict:
    """Create a sentiment row from pipeline result."""
    label_raw = result.get("label", "error").lower()
    label = label_map.get(label_raw, "error")
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
    else:
        # mean
        agg = df.groupby("label")["score"].mean().to_dict()

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
