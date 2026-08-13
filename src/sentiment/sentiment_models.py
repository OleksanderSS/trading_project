# src/sentiment/sentiment_models.py

import hashlib
from pathlib import Path

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_FINBERT_PIPELINE = None
_TOKENIZER = None
_CACHE: dict[str, dict[str, str]] = {}  # result cache by text hash

#: Scoring 15,274 articles takes over half an hour of FinBERT on CPU, and
#: `_CACHE` dies with the process — so every rebuild paid it again, in full,
#: for articles it had already read. The corpus grows by a few hundred a day
#: and the rest is identical, so almost all of that work is repeated.
#:
#: Keyed by model name as well as text: a different model is a different
#: answer, and a cache that forgets which one produced a row would serve
#: FinBERT's verdicts for whatever replaces it.
_CACHE_PATH = Path("data/cache/sentiment/finbert_scores.parquet")
_CACHE_MODEL = "ProsusAI/finbert"
_CACHE_LOADED = False
_CACHE_DIRTY = False
#: Rows scored since the last write. A crash costs at most this many.
_SAVE_EVERY = 500
_CACHE_SAVED_SIZE = 0


def _last_saved_size() -> int:
    return _CACHE_SAVED_SIZE


def _stable_hash(text: str) -> str:
    """Short hash of text for caching."""
    return hashlib.sha256(
        f"{_CACHE_MODEL}\x00{text}".encode()
    ).hexdigest()[:16]


def load_sentiment_cache() -> int:
    """Read previously scored texts from disk. Returns how many were loaded."""
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return len(_CACHE)
    _CACHE_LOADED = True
    if not _CACHE_PATH.exists():
        return 0
    try:
        stored = pd.read_parquet(_CACHE_PATH)
    except Exception as exc:                      # noqa: BLE001 - cache only
        logger.warning(
            "Could not read the sentiment cache (%s: %s); rescoring from "
            "scratch this run.", type(exc).__name__, exc,
        )
        return 0

    for row in stored.to_dict("records"):
        key = row.get("hash")
        if key:
            _CACHE[key] = {"text": row.get("text", ""),
                           "label": row.get("label", "neutral"),
                           "score": float(row.get("score", 0.0))}
    globals()['_CACHE_SAVED_SIZE'] = len(_CACHE)
    logger.info("Sentiment cache: %d texts already scored.", len(_CACHE))
    return len(_CACHE)


def save_sentiment_cache() -> int:
    """Persist newly scored texts. Returns the number of rows written."""
    global _CACHE_DIRTY
    if not _CACHE_DIRTY or not _CACHE:
        return 0
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [{"hash": key, "text": value.get("text", ""),
              "label": value.get("label", "neutral"),
              "score": float(value.get("score", 0.0))}
             for key, value in _CACHE.items()]
        )
        # Write beside the target and replace, so a crash mid-write cannot
        # leave a truncated cache that reads as "nothing scored yet".
        temporary = _CACHE_PATH.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(_CACHE_PATH)
        _CACHE_DIRTY = False
        globals()['_CACHE_SAVED_SIZE'] = len(_CACHE)
        logger.info("Sentiment cache: %d texts stored.", len(frame))
        return len(frame)
    except Exception as exc:                      # noqa: BLE001 - cache only
        logger.warning(
            "Could not write the sentiment cache (%s: %s); this run's scores "
            "will have to be recomputed next time.", type(exc).__name__, exc,
        )
        return 0

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

    load_sentiment_cache()

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

        # Save as we go, not only at the end. Scoring this project's corpus
        # takes over two hours on CPU, and a run stopped part-way -- which
        # has happened three times in one day, between a Windows update and
        # two deliberate restarts -- lost every score it had computed. The
        # write is a few hundred kilobytes against ten minutes of work.
        if len(_CACHE) - _last_saved_size() >= _SAVE_EVERY:
            save_sentiment_cache()

    save_sentiment_cache()
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
            globals()['_CACHE_DIRTY'] = True
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
