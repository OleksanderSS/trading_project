# collectors/hf_collector.py

import pandas as pd
from huggingface_hub import HfApi
from collectors.base_collector import BaseCollector
from enrichment.sentiment_analyzer import SentimentEnricher
from enrichment.keyword_extractor import KeywordExtractor
from utils.cache_utils import CacheManager
from utils.news_harmonizer import harmonize_batch
from typing import List, Dict, Optional
import logging
from config.secrets_manager import Secrets

logger = logging.getLogger("trading_project.hf_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False

secrets = Secrets()


class HFCollector(BaseCollector):
    def __init__(
        self,
        token: str = None,
        queries: Optional[List[str]] = None,
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        limit: int = 25,
        table_name: str = "hf_models",
        db_path: str = ":memory:",
        cache_path: Optional[str] = None,
        strict: bool = True,
        **kwargs
    ):
        cache_path = cache_path or "./data/cache"
        cache = CacheManager(base_path=cache_path)

        super().__init__(db_path=db_path, table_name=table_name, strict=strict, **kwargs)

        self.api = HfApi(token=token or secrets.get("HF_TOKEN"))
        self.queries = queries or [
            "finance", "trading", "stock", "sentiment", "quant", "time-series",
            "market", "macro", "econometrics", "crypto", "financial-news", "forecasting"
        ]
        self.limit = limit
        self.cache_manager = cache
        self.keyword_dict = keyword_dict or {}
        self.analyzer = SentimentEnricher(cache_manager=self.cache_manager, keyword_dict=self.keyword_dict)

        logger.info("[HFCollector] [OK] Initialized")

    def fetch_models(self) -> pd.DataFrame:
        all_models = []
        for q in self.queries:
            try:
                models = list(self.api.list_models(search=q, limit=self.limit))
                all_models.extend(models)
                logger.info(f"[HFCollector] [SEARCH] {len(models)} моwhereлей for forпиту '{q}'")
            except Exception as e:
                logger.error(f"[HFCollector] [ERROR] Error forпиту '{q}': {e}")

        # Дедуплandкацandя
        unique_models = {m.modelId: m for m in all_models}.values()
        logger.info(f"[HFCollector] [SEARCH] Унandкальних моwhereлей: {len(unique_models)}")

        entries = []
        for m in unique_models:
            # Беwithпечна обробка cardData
            card_data = getattr(m, "cardData", None) or {}
            description = card_data.get("description", m.modelId)
            pipeline_tag = getattr(m, "pipeline_tag", None)
            downloads = getattr(m, "downloads", None)
            likes = getattr(m, "likes", None)
            last_modified = getattr(m, "lastModified", None)

            # Постфandльтри якостand (мякand пороги)
            if pipeline_tag is None:
                continue
            if (downloads is not None and downloads < 20) and (likes is not None and likes < 5):
                continue

            entries.append({
                "title": m.modelId,
                "description": description,
                "summary": pipeline_tag,
                "published_at": pd.Timestamp.utcnow().normalize().isoformat(),
                "url": f"https://huggingface.co/{m.modelId}",
                "type": "qualitative",
                "source": "HuggingFace",
                "value": None,
                "sentiment": None,
                "result": None,
                "ticker": "GENERAL",
                "raw_data_fields": {
                    "modelId": m.modelId,
                    "pipeline_tag": pipeline_tag,
                    "downloads": downloads,
                    "likes": likes,
                    "lastModified": str(last_modified),
                }
            })

        df = pd.DataFrame(harmonize_batch(entries, source="HuggingFace"))
        logger.info(f"[HFCollector] [SEARCH] Пandсля гармонandforцandї: {len(df)} forписandв")

        if df.empty:
            logger.warning("[HFCollector] [WARN] Жодна model not пройшла гармонandforцandю")
            return df

        # Keywords (опцandонально)
        if self.keyword_dict:
            extractor = KeywordExtractor(self.keyword_dict)
            df["keywords"] = df["description"].astype(str).apply(extractor.extract_keywords)
            df["match_count"] = df["keywords"].apply(len)
            df = df[df["match_count"] > 0].reset_index(drop=True)
            logger.info(f"[HFCollector] [SEARCH] Пandсля фandльтрацandї по ключовим словам: {len(df)}")

        # Сентимент (опцandонально for описandв)
        df = self.analyzer.analyze_sentiment(df, text_col="description")
        df["sentiment"] = df.get("news_score", pd.NA)
        df.drop(columns=["news_score"], errors="ignore", inplace=True)

        return df

    def collect(self) -> pd.DataFrame:
        df = self.fetch_models()
        if not df.empty:
            self._save_batch(df)
        return df

    def fetch(self) -> pd.DataFrame:
        return self.collect()

    def _save_batch(self, df: pd.DataFrame):
        if df.empty:
            return
        records = df.to_dict(orient="records")
        self.save(records, strict=self.strict)