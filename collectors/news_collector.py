# collectors/news_collector.py

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from collectors.base_collector import BaseCollector
from collectors.rss_collector import RSSCollector
from enrichment.sentiment_analyzer import SentimentEnricher
from enrichment.keyword_extractor import KeywordExtractor
from config.config_loader import load_yaml_config
from utils.cache_utils import CacheManager
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("trading_project.news_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


class NewsCollector(BaseCollector):
    def __init__(
        self,
        rss_feeds: Optional[Dict[str, str]] = None,
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        db_path: str = ":memory:",
        table_name: str = "news_data",
        config_path: Optional[str] = None,
        strict: bool = True,
        cache_manager: Optional[CacheManager] = None,
        limit: int = 50
    ):
        base_dir = os.path.dirname(__file__)
        config_path = config_path or os.path.join(base_dir, "..", "config", "news_sources.yaml")
        config = load_yaml_config(config_path)

        self.rss_feeds = rss_feeds or config.get("rss", {})
        self.keyword_dict = keyword_dict if keyword_dict is not None else config.get("keywords", {})

        if cache_manager is None:
            cache_path = os.path.join(base_dir, "..", "data", "cache")
            os.makedirs(cache_path, exist_ok=True)
            cache_manager = CacheManager(base_path=cache_path)

        self.analyzer = SentimentEnricher(
            cache_manager=cache_manager,
            keyword_dict=self.keyword_dict
        )

        self.rss_collector = RSSCollector(
            rss_feeds=self.rss_feeds,
            keyword_dict=self.keyword_dict,
            analyzer=self.analyzer,
            cache_manager=cache_manager,
            db_path=db_path,
            table_name=table_name,
            strict=strict,
            limit=limit
        )

        # Викликаємо баwithовий конструктор тandльки with тим, що вandн реально приймає
        super().__init__(db_path=db_path, table_name=table_name, strict=strict)

        logger.info(f"[NewsCollector] Initialized with {len(self.keyword_dict)} keyword groups and {len(self.rss_feeds)} RSS feeds")

    def filter_by_keywords(self, df: pd.DataFrame, text_col: str = "description", min_matches: int = 2) -> pd.DataFrame:
        if df.empty or text_col not in df.columns:
            return df
        extractor = KeywordExtractor(self.keyword_dict)
        df["keywords"] = df[text_col].astype(str).apply(extractor.extract_keywords)
        df["match_count"] = df["keywords"].apply(len)
        return df[df["match_count"] >= 1].reset_index(drop=True)  

    def filter_similar_news(self,
        df: pd.DataFrame,
        text_col: str = "description",
        threshold: float = 0.9) -> pd.DataFrame:
        if df.empty or text_col not in df.columns:
            return df
        texts = df[text_col].astype(str).tolist()
        vectorizer = TfidfVectorizer().fit_transform(texts)
        sim_matrix = cosine_similarity(vectorizer)
        to_drop = set()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if sim_matrix[i, j] > threshold:
                    to_drop.add(j)
        return df.drop(df.index[list(to_drop)]).reset_index(drop=True)

    def fetch(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        df_list = []

        for source, url in self.rss_feeds.items():
            df = self.rss_collector.fetch_single(source, url)
            logger.info(f"[NewsCollector] [SEARCH] {source}: received {len(df)} forписandв")

            if not df.empty:
                # [OK] ВИПРАВЛЕНО - перевіряємо наявність колонки перед обробкою
                if "published_at" not in df.columns:
                    logger.warning(f"[NewsCollector] {source}: no 'published_at' column")
                    continue
                    
                #  Приводимо published_at до Timestamp беwith andймwithони
                df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
                
                # [OK] ВИПРАВЛЕНО - перевіряємо на None перед доступом до атрибутів
                if df["published_at"].notna().any():
                    if df["published_at"].dt.tz is not None:
                        df["published_at"] = df["published_at"].dt.tz_convert(None)
                else:
                    logger.warning(f"[NewsCollector] {source}: all published_at are NaT")
                    continue

                if start_date:
                    start_date = pd.to_datetime(start_date, errors="coerce")
                    # Ensure timezone-naive for consistency
                    if hasattr(start_date, 'tz') and start_date.tz is not None:
                        start_date = start_date.tz_convert(None)  # ВИПРАВЛЕНО
                    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
                    df = df[df["published_at"] >= start_date]
                if end_date:
                    end_date = pd.to_datetime(end_date, errors="coerce")
                    # Ensure timezone-naive for consistency
                    if hasattr(end_date, 'tz') and end_date.tz is not None:
                        end_date = end_date.tz_convert(None)  # ВИПРАВЛЕНО
                    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
                    df = df[df["published_at"] <= end_date]

                logger.info(f"[NewsCollector]  {source}: пandсля фandльтрацandї по даandх forлишилось {len(df)} forписandв")
                if not df.empty:
                    df_list.append(df)

        if not df_list:
            logger.warning("[NewsCollector] [ERROR] Жодnot джерело not повернуло данand")
            return pd.DataFrame()

        df_all = pd.concat(df_list, ignore_index=True)
        logger.info(f"[NewsCollector]  Отримано {len(df_all)} forписandв до фandльтрацandї")

        df_all = self.filter_by_keywords(df_all)
        logger.info(f"[NewsCollector] [SEARCH] Пandсля фandльтрацandї по ключовим словам forлишилось {len(df_all)} forписandв")

        df_all = self.filter_similar_news(df_all, text_col="description", threshold=0.7)
        logger.info(f"[NewsCollector] [BRAIN] Пandсля фandльтрацandї схожих forлишилось {len(df_all)} forписandв")

        df_all = self.analyzer.analyze_sentiment(df_all, text_col="description")
        df_all["ticker"] = "GENERAL"

        self._save_batch(df_all)
        logger.info(f"[NewsCollector] [OK] Збережено {len(df_all)} forписandв with {len(self.rss_feeds)} джерел")
        return df_all

    def _save_batch(self, df: pd.DataFrame):
        if df.empty:
            return
        records = []
        for _, row in df.iterrows():
            published_at = row.get("published_at")
            
            # Skip rows with no date - don't add fake dates
            if pd.isna(published_at):
                continue  # Skip instead of adding fake date
                
            records.append({
                "title": row.get("title", "No Title"),
                "description": row.get("description", ""),
                "summary": row.get("summary", ""),
                "published_at": published_at,
                "url": row.get("url", "N/A"),
                "type": row.get("type", "qualitative"),
                "source": row.get("source", "RSS"),
                "ticker": row.get("ticker", "GENERAL"),
                "raw_data_fields": row.to_dict()
            })
        self.save(records, strict=self.strict)

    def collect(self) -> pd.DataFrame:
        """Збирає новини за останні 30 днів без тікерів - фільтрує по словниках"""
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # ВИПРАВЛЕНО: 30 днів замість 7
        
        logger.info(f"[NewsCollector] [START] Збір новин за останні 30 днів")
        return self.fetch(start_date=start_date, end_date=end_date)

    def collect_with_tickers(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Збирає новини для конкретних тікерів (старий метод)"""
        logger.info(f"[NewsCollector] [START] Збір новин for {tickers} with {start_date.date()} по {end_date.date()}")
        return self.fetch(start_date=start_date, end_date=end_date)