# collectors/rss_collector.py

import feedparser
import pandas as pd
from typing import List, Dict, Optional, Any
from collectors.base_collector import BaseCollector
from enrichment.sentiment_analyzer import SentimentEnricher
from enrichment.keyword_extractor import KeywordExtractor
from utils.cache_utils import CacheManager
from dateutil import parser
import logging
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("rss_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False

def parse_date_safe(date_str):
    if not date_str:
        return pd.Timestamp.now().normalize() # Даємо хоча б сьогоднішню дату
    
    # Список полів, які ми могли отримати з entry
    # Якщо прийшов словник, спробуємо знайти альтернативи
    raw_val = date_str
    
    try:
        # Спроба №1: Стандартний pandas
        dt = pd.to_datetime(raw_val, errors='coerce')
        
        # Спроба №2: Якщо pandas видав NaT, використовуємо parser
        if pd.isna(dt):
            dt = parser.parse(str(raw_val))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = pd.to_datetime(dt)
            
        # ПРИНЦИПОВО: Робимо всі дати tz-naive (без часової зони), 
        # бо твій YFCollector теж так робить. Це дозволить їх порівнювати.
        if dt.tzinfo is not None:
            dt = dt.tz_localize(None)
            
        return dt
    except Exception:
        # Якщо все провалилося - не повертаємо None! 
        # Повертаємо поточний час, щоб новина потрапила в базу.
        return pd.Timestamp.now().normalize()

class RSSCollector(BaseCollector):
    def __init__(self,
                 rss_feeds: Dict[str, str],
                 keyword_dict: Optional[Dict[str, List[str]]] = None,
                 limit: Optional[int] = 50,
                 table_name: str = "rss_data",
                 db_path: str = ":memory:",
                 strict: bool = True,
                 cache_manager: Optional[CacheManager] = None,
                 analyzer: Optional[SentimentEnricher] = None):

        super().__init__(db_path=db_path, table_name=table_name, strict=strict)

        self.rss_feeds = rss_feeds
        self.limit = limit
        self.keyword_dict = keyword_dict or {}
        # Use only relevant keyword categories for RSS filtering
        relevant_categories = ["market_terms", "finance_economy", "sectors_tech", "professional_indicators"]
        self.flat_keywords = []
        for category in relevant_categories:
            if category in self.keyword_dict:
                self.flat_keywords.extend(self.keyword_dict[category])
        
        logger.info(f"[RSSCollector] Ініціалізовано з {len(self.flat_keywords)} релевантними ключовими словами")
        self.cache_manager = cache_manager or CacheManager(base_path="./data/cache")
        self.analyzer = analyzer

    def _parse_rss(self, source: str, url: str) -> List[Dict[str, Any]]:
        """Parse RSS feed with fast pre-filtering"""
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            if getattr(feed, "bozo", False):
                logger.warning(f"[RSS] Некоректний RSS-фід: {url}")
                return []

            articles = []
            for idx, entry in enumerate(feed.entries):
                if self.limit and idx >= self.limit:
                    break

                # 🚀 ШВИДКА ПРЕ-ФІЛЬТРАЦІЯ до обробки
                if not self._is_relevant_content(entry):
                    continue

                pub_str = entry.get("published") or entry.get("updated")
                pub_dt = parse_date_safe(pub_str)
                
                article = {
                    "published_at": pub_dt,
                    "description": entry.get('description', ''),
                    "title": entry.get('title', ''),
                    "source": source,
                    "url": entry.get('link', ''),
                    "summary": entry.get('summary', ''),
                }
                
                articles.append(article)
            
            logger.info(f"[RSSCollector] {source}: {len(articles)} релевантних статей з {len(feed.entries)} загалом")
            return articles
            
        except Exception as e:
            logger.error(f"[RSSCollector] Error parsing {source}: {e}")
            return []
    
    def _is_relevant_content(self, entry) -> bool:
        """Швидка перевірка релевантності контенту"""
        title = (entry.get('title', '') + ' ' + entry.get('description', '')).lower()
        
        # Швидкі фільтри нерелевантного контенту
        blocked_keywords = [
            'nfl', 'sports', 'game', 'player', 'team', 'coach',
            'celebrity', 'entertainment', 'movie', 'music', 'concert',
            'recipe', 'food', 'travel', 'vacation', 'fashion', 'style',
            'gaming', 'esports', 'streaming', 'tv show', 'reality tv'
            # Видаляємо 'netflix' - це фінансова компанія, а не розвага!
        ]
        
        # Якщо є заблоковані слова - пропускаємо
        if any(blocked in title for blocked in blocked_keywords):
            return False
        
        # Перевіряємо наявність релевантних фінансових слів
        relevant_keywords = [
            'market', 'stock', 'trade', 'economy', 'financial', 'business',
            'earnings', 'revenue', 'profit', 'investment', 'fed', 'inflation',
            'interest', 'gdp', 'recession', 'merger', 'acquisition', 'tech',
            'ai', 'semiconductor', 'energy', 'oil', 'commodity', 'currency'
        ]
        
        # Якщо є хоча б одне релевантне слово - беремо
        return any(relevant in title for relevant in relevant_keywords)

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if self.analyzer is None:
            # Без аналізу - просто повертаємо DataFrame
            df["sentiment"] = None
            return df
        df["description"] = df["description"].astype(str).fillna("").str.strip()
        enriched = self.analyzer.analyze_sentiment(df, text_col="description")
        if not isinstance(enriched, pd.DataFrame):
            logger.error("[RSS] analyze_sentiment повернув не DataFrame")
            return df
        enriched["sentiment"] = enriched.get("news_score", pd.NA)
        return enriched

    def _finalize(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        # Filter out articles with no date first
        valid_articles = [a for a in articles if a.get("published_at") is not None]
        
        if not valid_articles:
            logger.warning("[RSS] Всі статті мають відсутні дати")
            return pd.DataFrame(columns=["published_at", "description", "type", "value", "sentiment", "source", "url"])
            
        df_raw = pd.DataFrame(valid_articles)
        if df_raw.empty:
            logger.warning("[RSS] Порожній DataFrame після парсингу")
            return pd.DataFrame(columns=["published_at", "description", "type", "value", "sentiment", "source", "url"])

        df_enriched = self._enrich(df_raw)
        df_keywords = self.filter_by_keywords(df_enriched, text_col="description")
        df_unique = self.filter_similar_news(df_keywords, text_col="description", threshold=0.9)

        # Add missing columns
        df_unique['type'] = None
        df_unique['value'] = None

        df_final = df_unique[["published_at", "description", "type", "value", "sentiment", "source", "url"]].copy()

        df_final["published_at"] = pd.to_datetime(df_final["published_at"], errors="coerce")

        invalid_dates = df_final["published_at"].isna().sum()
        total_rows = len(df_final)

        if invalid_dates == total_rows:
            logger.warning("[RSSCollector] ⚠️ Усі published_at = NaT — перевір формат дати у RSS")
        elif invalid_dates > 0:
            logger.warning(f"[RSSCollector] ⚠️ {invalid_dates} з {total_rows} записів мають NaT у published_at")
        else:
            logger.info(f"[RSSCollector] ✅ Всі {total_rows} записів мають валідні дати")

        logger.info(f"[RSSCollector] ✅ Залишилось {len(df_final)} релевантних записів")
        return df_final

    def fetch(self, start_date=None, end_date=None) -> pd.DataFrame:
        all_articles = []
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            # Ensure timezone-naive for consistency
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            # Ensure timezone-naive for consistency
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)

        for source, url in self.rss_feeds.items():
            articles = self._parse_rss(source, url)
            if start_date:
                articles = [a for a in articles if a["published_at"] and a["published_at"] >= start_date]
            if end_date:
                articles = [a for a in articles if a["published_at"] and a["published_at"] <= end_date]
            all_articles.extend(articles)

        return self._finalize(all_articles)

    def fetch_single(self, source: str, url: str) -> pd.DataFrame:
        articles = self._parse_rss(source, url)
        return self._finalize(articles)

    def collect(self) -> List[Dict[str, Any]]:
        df = self.fetch()
        if not isinstance(df, pd.DataFrame):
            logger.error("[RSS] fetch повернув не DataFrame")
            return []
        records = df.to_dict(orient="records")
        if records:
            self.save(records, strict=False)
        return records

    def filter_by_keywords(self, df: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
        if df.empty or text_col not in df.columns:
            return df
        extractor = KeywordExtractor(self.keyword_dict)
        df["keywords"] = df[text_col].astype(str).apply(extractor.extract_keywords)
        df["match_count"] = df["keywords"].apply(len)
        return df[df["match_count"] > 0].reset_index(drop=True)

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
