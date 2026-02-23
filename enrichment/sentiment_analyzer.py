# enrichment/sentiment_analyzer.py

from enrichment.keyword_extractor import KeywordExtractor
from utils.logger_fixed import ProjectLogger
from utils.cache_utils import CacheManager
from utils.sentiment_core import get_model, make_sentiment_key, compute_news_score_safe
from utils.mention_utils import safe_get
from transformers import AutoTokenizer
import torch
import pandas as pd
import time
from typing import Optional, List, Dict


logger = ProjectLogger.get_logger("SentimentEnricher")


class SentimentEnricher:
    def __init__(
        self,
        cache_manager: CacheManager,
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        batch_delay: float = 0.05
    ):
        self.cache_manager = cache_manager
        self.batch_delay = batch_delay
        
        # Initialize logger with English message
        logger.debug("[OK] Logger started [SentimentEnricher]")
        
        # Include fast model for sentiment analysis
        try:
            from utils.sentiment_core import get_model
            from transformers import AutoTokenizer
            self.sentiment_model = get_model()
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            logger.info("[SentimentEnricher] Sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"[SentimentEnricher] [WARN] Failed to load sentiment model: {e}")
            self.sentiment_model = None
            self.tokenizer = None

        #  Автоматичний вибір batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_batch_size = 32 if self.device == "cuda" else 8

        if not isinstance(keyword_dict, dict):
            keyword_dict = {}

        self.keyword_dict = keyword_dict
        self.keyword_extractor = KeywordExtractor(keywords=keyword_dict)

        if not self.keyword_extractor.keywords and not self.keyword_extractor.tickers:
            logger.warning("[SentimentEnricher] [WARN] Keywords not provided - filtering will be empty")
        else:
            logger.debug(f"[SentimentEnricher] [SEARCH] Keywords: {len(self.keyword_extractor.keywords)}")
            logger.debug(f"[SentimentEnricher] [TARGET] Tickers: {len(self.keyword_extractor.tickers)}")
            logger.debug(f"[SentimentEnricher] [TOP5] Top-5 words: {self.keyword_extractor.keywords[:5]}")

    def trim_to_max_tokens(self, text, max_tokens=512):
        text = "" if text is None else str(text)
        if self.tokenizer is None:
            return text[:1000]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            padding=False,
            return_tensors=None
        )
        return self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    def analyze_sentiment(self, df: pd.DataFrame, text_col: str = "description", batch_size: int = 16) -> pd.DataFrame:
        if df.empty or text_col not in df.columns:
            return df

        df["trimmed_text"] = df[text_col].astype(str).fillna('').apply(self.trim_to_max_tokens)
        empty_mask = df["trimmed_text"].str.strip().eq("")
        source_types = df["source_type"].tolist() if "source_type" in df.columns else [None] * len(df)

        df["sentiment_key"] = None
        if empty_mask.any():
            empty_keys = [f"sent_empty_{i}" for i in df.index[empty_mask]]
            df.loc[empty_mask, "sentiment_key"] = empty_keys
        if (~empty_mask).any():
            df.loc[~empty_mask, "sentiment_key"] = df.loc[~empty_mask, "trimmed_text"].apply(make_sentiment_key)

        rows, new_items = [], []

        # Перевandрка кешу
        for t, key, is_empty, source_type in zip(df["trimmed_text"], df["sentiment_key"], empty_mask, source_types):
            if is_empty:
                rows.append({
                    "text": t,
                    "sentiment_key": key,
                    "label": "neutral",
                    "score": 0.0,
                    "keywords": [],
                    "match_count": 0,
                    "news_score": 0.0,
                    "parsed_at": pd.to_datetime('now'),
                    "source_type": source_type
                })
                continue

            cached = self.cache_manager.load_pickle(key)
            if cached:
                cached["sentiment_key"] = key
                if "published_at" not in cached:
                    cached["parsed_at"] = pd.to_datetime('now')
                rows.append(cached)
            else:
                new_items.append((t, source_type))

        # Обробка нових текстandв батчами
        if self.sentiment_model is None or self.tokenizer is None:
            # Якщо моwhereлand вandдключенand, поверandємо neutral sentiment
            for t, source_type in new_items:
                keywords = self.keyword_extractor.extract_keywords(t)
                rows.append({
                    "text": t,
                    "sentiment_key": make_sentiment_key(t),
                    "label": "neutral",
                    "score": 0.0,
                    "keywords": keywords,
                    "match_count": len(keywords),
                    "news_score": compute_news_score_safe("neutral", 0.0, keywords),
                    "parsed_at": pd.to_datetime('now'),
                    "source_type": source_type
                })
        else:
            for i in range(0, len(new_items), batch_size):
                batch_items = new_items[i:i + batch_size]
                batch = [t for t, _ in batch_items]
                batch_source_types = [st for _, st in batch_items]

                try:
                    # Токенandforцandя with усandченням
                    encoded_batch = self.tokenizer(
                        batch,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_tensors="pt"
                    )

                    # Прогandн череwith model
                    with torch.no_grad():
                        outputs = self.sentiment_model(**encoded_batch)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)

                    # Перетворення реwithульandтandв у label + score
                    labels = probs.argmax(dim=-1).tolist()
                    scores = probs.max(dim=-1).values.tolist()

                    id2label = {0: "negative", 1: "neutral", 2: "positive"}

                    for t, source_type, label_id, score in zip(batch, batch_source_types, labels, scores):
                        label = id2label.get(label_id, "neutral")

                        keywords = self.keyword_extractor.extract_keywords(t)
                        news_score = compute_news_score_safe(label, score, keywords)

                        row = {
                            "text": t,
                            "sentiment_key": make_sentiment_key(t),
                            "label": label,
                            "score": float(score),
                            "keywords": keywords,
                            "match_count": len(keywords),
                            "news_score": news_score,
                            "parsed_at": pd.to_datetime('now'),
                            "source_type": source_type
                        }
                        rows.append(row)
                        self.cache_manager.save_pickle(row["sentiment_key"], row)

                except Exception as e:
                    logger.exception(f"[SentimentEnricher] FinBERT batch error: {e}")
                    for t, source_type in zip(batch, batch_source_types):
                        rows.append({
                            "text": t,
                            "sentiment_key": make_sentiment_key(t),
                            "label": "neutral",
                            "score": 0.0,
                            "keywords": [],
                            "match_count": 0,
                            "news_score": 0.0,
                            "parsed_at": pd.to_datetime('now'),
                            "source_type": source_type
                        })

                time.sleep(self.batch_delay)

        # Формування реwithульandтandв
        text_to_row = {r["sentiment_key"]: r for r in rows}

        df["sentiment"] = df["sentiment_key"].apply(lambda h: safe_get(text_to_row, h, "label", default="neutral"))
        df["sentiment_score"] = df["sentiment_key"].apply(lambda h: safe_get(text_to_row, h, "score", default=0.0))
        df["keywords"] = df["sentiment_key"].apply(lambda h: safe_get(text_to_row, h, "keywords", default=[]))
        df["news_score"] = df["sentiment_key"].apply(lambda h: safe_get(text_to_row, h, "news_score", default=0.0))
        df["match_count"] = df["keywords"].apply(len)

        df.drop(columns=["sentiment_key", "trimmed_text"], inplace=True)
        logger.info(
            f"[SentimentEnricher] [OK] Аналandwith forвершено: "
            f"positive={sum(df['sentiment'] == 'positive')}, "
            f"negative={sum(df['sentiment'] == 'negative')}, "
            f"neutral={sum(df['sentiment'] == 'neutral')}"
        )
        return df
