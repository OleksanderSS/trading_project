# src/data/collectors/google_news_collector.py

import asyncio
import hashlib
import pandas as pd
import httpx
from gnews import GNews
from typing import List, Optional, Dict, Any

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager


class GoogleNewsCollector(BaseCollector):
    """Collector for Google News. Без resolve redirects — зберігаємо оригінальний URL."""
    collector_type = "google_news"

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

        params = self.configs.get("params", {})
        period = params.get("period", "7d")          # Скорочено з 60d → 7d за замовчуванням
        max_results = params.get("max_results", 10)   # Скорочено з 100 → 10 за замовчуванням

        self.api = GNews(period=period, max_results=max_results)
        self.delay = self.configs.get("delay", 0.5)
        self.max_concurrent = self.configs.get("max_concurrent_terms", 5)

        # Фільтри якості
        filter_cfg = self.configs.get("filter", {})
        self.min_source_quality = filter_cfg.get("min_source_quality", 0.0)
        self.exclude_title_keywords = [
            kw.lower() for kw in filter_cfg.get("exclude_title_keywords", [])
        ]
        self.require_keywords_in_title = filter_cfg.get("require_keywords_in_title", False)

        self.logger.info(
            f"GoogleNewsCollector: period={period}, max_results={max_results}, "
            f"max_concurrent={self.max_concurrent}, "
            f"min_source_quality={self.min_source_quality}"
        )

    async def run(
        self,
        tickers: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        search_terms: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Збирає новини, фільтрує через кеш та БД."""
        # Приймаємо і search_terms і keywords
        keywords_list = list(keywords.keys()) if isinstance(keywords, dict) else (keywords or [])
        all_terms = list(set(
            (tickers or []) + keywords_list + (search_terms or [])
        ))

        if not all_terms:
            self.logger.warning("GoogleNewsCollector: немає термінів для пошуку.")
            return None

        table_name = self.configs.get("table_name", "google_news")
        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"terms": sorted(all_terms)}

        # 1. Кеш
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info("[GoogleNews] Cache hit — нових статей немає.")
                        return None
                    return new_from_cache

        self.logger.info(f"[GoogleNews] Збір для {len(all_terms)} термінів (паралельно по {self.max_concurrent})...")

        # 2. Паралельний збір з семафором
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._fetch_with_semaphore(term, semaphore) for term in all_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.error(f"Помилка для терміну '{all_terms[i]}': {res}")
            elif res:
                all_articles.extend(res)

        if not all_articles:
            self.logger.info("[GoogleNews] Статей не знайдено.")
            return None

        # 3. Дедуплікація по URL ще до БД
        seen_urls = set()
        unique_articles = []
        for a in all_articles:
            url = a.get("link", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(a)

        df = pd.DataFrame(unique_articles)
        df["hash"] = df["link"].apply(
            lambda url: hashlib.sha256(str(url).encode()).hexdigest()
        )

        # 4. Перевірка кешу по хешу (швидко, без БД)
        if self.cache_manager:
            is_new = df["hash"].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("[GoogleNews] Всі статті вже в кеші.")
                return None

        # 5. Фільтрація через БД
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[GoogleNews] Нових статей не знайдено в БД.")
            if self.cache_manager:
                for h in df["hash"]:
                    self.cache_manager.set(h, True, ttl=86400)
                self.cache_manager.set(cache_key, df.to_dict("records"), cache_params, namespace="collectors")
            return None

        # 6. Збереження
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            for h in new_df["hash"]:
                self.cache_manager.set(h, True, ttl=86400)
            self.cache_manager.set(cache_key, df.to_dict("records"), cache_params, namespace="collectors")

        self.logger.info(f"[GoogleNews] Збережено {len(new_df)} нових статей.")
        return new_df

    async def _fetch_with_semaphore(self, term: str, semaphore: asyncio.Semaphore) -> List[Dict]:
        async with semaphore:
            result = await self._fetch_articles_for_term(term)
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            return result

    async def _fetch_articles_for_term(self, term: str) -> List[Dict]:
        """Завантажує статті для одного терміну. БЕЗ resolve redirects."""
        try:
            loop = asyncio.get_running_loop()
            news = await loop.run_in_executor(None, self.api.get_news, term)
            if not news:
                return []

            articles = []
            for entry in news:
                processed = self._process_entry(entry)
                if processed:
                    articles.append(processed)
            return articles

        except Exception as e:
            self.logger.error(f"Помилка збору новин для '{term}': {e}")
            return []

    def _process_entry(self, entry: Dict) -> Optional[Dict]:
        url = entry.get("url")
        if not url:
            return None

        title = entry.get("title", "") or ""
        source = entry.get("publisher", {}).get("title", "") or ""

        # Фільтр по стоп-словах в заголовку
        title_lower = title.lower()
        if any(kw in title_lower for kw in self.exclude_title_keywords):
            return None

        return {
            "title": title,
            "link": url,
            "published_date": pd.to_datetime(entry.get("published date"), utc=True, errors="coerce"),
            "source": source,
            "content": entry.get("description"),
        }