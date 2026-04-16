# src/data/collectors/rss_collector.py

import asyncio
import hashlib
import pandas as pd
import feedparser
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone

from .base_collector import BaseCollector
from src.core.logging.logger import ProjectLogger
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager


class RSSCollector(BaseCollector):
    """Collector for fetching news from RSS feeds."""
    collector_type = "rss"
    data_type = "news"

    def __init__(
        self,
        configs: Dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: Optional[CacheManager] = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.logger = ProjectLogger.get_logger(__name__)
        self.period_days = self._parse_period_to_days()

        # Фільтри якості з конфігу
        filter_cfg = self.configs.get("filter", {})
        self.min_source_quality = filter_cfg.get("min_source_quality", 0.0)
        self.exclude_title_keywords = [
            kw.lower() for kw in filter_cfg.get("exclude_title_keywords", [])
        ]

        # quality_weights з knowledge_base передаємо через kwargs якщо є
        self._quality_weights: Dict[str, float] = kwargs.get("quality_weights", {})

        self.logger.info(
            f"RSSCollector initialized. Period: {self.period_days}d, "
            f"min_quality: {self.min_source_quality}"
        )

    def _parse_period_to_days(self) -> int:
        period_str = self.configs.get("params", {}).get("period", "7d")
        if "d" in period_str:
            return int(period_str.replace("d", ""))
        return 7

    async def run(
        self,
        tickers: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Збирає новини з RSS фідів, фільтрує, дедуплікує, зберігає."""
        table_name = self.configs.get("table_name", "rss_news")
        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"period_days": self.period_days}

        # 1. Кеш
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info("[RSS] Cache hit — нових статей немає.")
                        return None
                    return new_from_cache

        # 2. Завантажуємо фіди з конфігу (НЕ створюємо новий config_manager)
        # Фіди беремо з kwargs або з конфіга
        feeds = kwargs.get("rss_feeds") or self.configs.get("feeds", [])
        if not feeds:
            # Fallback: беремо з config_manager якщо переданий
            config_manager = kwargs.get("config_manager")
            if config_manager:
                kb = config_manager.get_config("knowledge_base")
                feeds = kb.get("rss_feeds", [])

        if not feeds:
            self.logger.warning("No RSS feeds configured. Skipping.")
            return None

        self.logger.info(f"[RSS] Fetching {len(feeds)} feeds...")

        # 3. Паралельний збір
        tasks = [self._fetch_feed(feed["name"], feed["url"]) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        flat_articles = []
        for res in results:
            if isinstance(res, list):
                flat_articles.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(f"Feed error: {res}")

        if not flat_articles:
            self.logger.info("[RSS] No articles found.")
            return None

        self.logger.info(f"[RSS] Collected {len(flat_articles)} raw articles.")

        df = pd.DataFrame(flat_articles)

        # 4. Hash для дедуплікації
        df["hash"] = df["link"].apply(
            lambda url: hashlib.sha256(str(url).encode()).hexdigest()
        )

        # 5. Кеш по хешу
        if self.cache_manager:
            is_new = df["hash"].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("[RSS] Всі статті вже в кеші.")
                return None

        # 6. Фільтрація через БД
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[RSS] Нових статей не знайдено в БД.")
            if self.cache_manager:
                for h in df["hash"]:
                    self.cache_manager.set(h, True, ttl=86400)
                self.cache_manager.set(
                    cache_key, df.to_dict("records"), cache_params, namespace="collectors"
                )
            return None

        # 7. Збереження
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            for h in new_df["hash"]:
                self.cache_manager.set(h, True, ttl=86400)
            self.cache_manager.set(
                cache_key, df.to_dict("records"), cache_params, namespace="collectors"
            )

        self.logger.info(f"[RSS] Збережено {len(new_df)} нових статей.")
        return new_df

    async def _fetch_feed(self, name: str, url: str) -> List[Dict]:
        """Завантажує один RSS фід і фільтрує статті."""
        try:
            client = self.http_client_factory.get_http_client()
            response = await client.get(url, timeout=self.configs.get("timeout", 20))
            response.raise_for_status()

            feed_data = feedparser.parse(response.text)
            limit = self.configs.get("params", {}).get("limit_per_feed", 20)
            # Використовуємо UTC для порівняння з UTC датами з RSS
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.period_days)

            # ЛОГУВАННЯ: Кількість записів у фіді
            self.logger.info(f"[RSS] Feed '{name}': {len(feed_data.entries)} entries found")

            articles = []
            skipped_no_date = 0
            skipped_old = 0
            skipped_filter = 0
            
            for i, entry in enumerate(feed_data.entries[:limit]):
                processed = self._process_entry(entry, name)
                
                if not processed:
                    skipped_filter += 1
                    continue
                
                if processed["published_date"] < cutoff:
                    skipped_old += 1
                    self.logger.debug(f"[RSS] Entry {i} skipped (too old: {processed['published_date']} < {cutoff})")
                    continue
                
                articles.append(processed)

            # ЛОГУВАННЯ: Результати фільтрації
            self.logger.info(
                f"[RSS] Feed '{name}': {len(articles)} articles after filtering "
                f"(skipped: {skipped_filter} filter, {skipped_old} old)"
            )
            
            return articles

        except Exception as e:
            self.logger.error(f"Error fetching feed '{name}' ({url}): {e}", exc_info=True)
            return []

    def _process_entry(self, entry: Dict, source_name: str) -> Optional[Dict]:
        """Обробляє один запис RSS з фільтрацією по якості."""
        published_str = entry.get("published")
        if not published_str:
            self.logger.debug(f"[RSS] Entry skipped: no 'published' field")
            return None

        try:
            # Парсуємо дату без .astimezone() - це викликає помилку
            published_date = pd.to_datetime(published_str, utc=True)
        except (ValueError, TypeError) as e:
            self.logger.debug(f"[RSS] Entry skipped: date parse error '{published_str}' - {e}")
            return None

        title = entry.get("title", "") or ""

        # Фільтр по стоп-словах
        title_lower = title.lower()
        if any(kw in title_lower for kw in self.exclude_title_keywords):
            self.logger.debug(f"[RSS] Entry skipped: excluded keyword in title '{title}'")
            return None

        # Фільтр по якості джерела
        if self.min_source_quality > 0 and self._quality_weights:
            source_lower = source_name.lower()
            quality = self._quality_weights.get(source_lower, self._quality_weights.get("default_weight", 0.3))
            if quality < self.min_source_quality:
                self.logger.debug(f"[RSS] Entry skipped: quality {quality} < {self.min_source_quality}")
                return None

        return {
            "title": title,
            "link": entry.get("link"),
            "published_date": published_date,
            "source": source_name,
            "content": entry.get("summary"),
        }