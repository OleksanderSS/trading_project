# models/sentiment_models.py

import hashlib
import pandas as pd
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

_FINBERT_PIPELINE = None
_TOKENIZER = None
_CACHE: Dict[str, Dict[str, str]] = {}  # кеш реwithульandтandв по хешу тексту

def _stable_hash(text: str) -> str:
    """Короткий хеш тексту for кешування"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def get_finbert_pipeline(device: int = None):
    """Поверandє глобальний пайплайн FinBERT. Заванandжує один раwith."""
    global _FINBERT_PIPELINE, _TOKENIZER
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE

    try:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        _TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _FINBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=_TOKENIZER,
            device=device
        )
        logger.info(f"[OK] FinBERT forванandжено ({'cuda' if device == 0 else 'cpu'})")
    except Exception as e:
        logger.exception(f"[ERROR] Error forванandження FinBERT: {e}")
        _FINBERT_PIPELINE = None

    return _FINBERT_PIPELINE

def analyze_sentiment(texts: List[str], batch_size: int = 16, device: int = None, **kwargs) -> pd.DataFrame:
    """
    Аналandwithує сентимент списку текстandв батчами.
    Використовує кешування for повторних текстandв.
    Поверandє DataFrame: text, label, score.
    """
    pipe = get_finbert_pipeline(device=device)
    if pipe is None:
        logger.warning("[WARN] FinBERT unavailable, all тексти будуть neutral")
        return pd.DataFrame([{"text": t, "label": "neutral", "score": 0.0} for t in texts])

    rows = []
    label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_safe = [t if t.strip() else "neutral" for t in batch]

        # Перевandрка кешу
        uncached, uncached_idx = [], []
        for idx, t in enumerate(batch):
            h = _stable_hash(t)
            if h in _CACHE:
                rows.append(_CACHE[h])
            else:
                uncached.append(batch_safe[idx])
                uncached_idx.append(idx)

        if not uncached:
            continue

        try:
            results = pipe(uncached, truncation=True, max_length=512, **kwargs)
            for idx, res in zip(uncached_idx, results):
                label_raw = res.get("label", "error").lower()
                label = label_map.get(label_raw, "error")
                score = float(res.get("score", 0.0))
                row = {"text": batch[idx], "label": label, "score": score}
                rows.append(row)
                _CACHE[_stable_hash(batch[idx])] = row
        except Exception as e:
            logger.error(f"[WARN] Batch {i} error: {e}", exc_info=True)
            for idx in uncached_idx:
                rows.append({"text": batch[idx], "label": "error", "score": 0.0})

    return pd.DataFrame(rows)

def aggregate_sentiment(df: pd.DataFrame, normalize: bool = True, method: str = "mean") -> Dict[str, float]:
    """
    Обчислює агрегований сентимент по allх новинах.
    method: "mean" | "sum" | "count"
    Якщо normalize=True, сума трьох категорandй = 1.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    if df.empty:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    if method == "sum":
        agg = df.groupby("label")["score"].sum().to_dict()
    elif method == "count":
        agg = df["label"].value_counts(normalize=False).to_dict()
    else:  # mean
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
        logger.warning("[WARN] Усand оцandнки сентименту = 0, поверandємо 0 for allх категорandй")

    return res