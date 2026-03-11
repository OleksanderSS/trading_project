# src/data/collectors/newsapi_collector.py

import asyncio
import hashlib
import os
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager


class NewsAPICollector(BaseCollector):
    """Колектор для збору новин з NewsAPI."""
    collector_type = "newsapi"
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
        self.base_url = self.configs.get("base_url", "https://newsapi.org/v2/everything")
        self.language = self.configs.get("language", "en")
        self.page_size = self.configs.get("page_size", 20)
        self.api_key_env = self.configs.get("api_key_env", "NEWS_API_KEY")
        self.hash_keys = self.configs.get("hash_keys", ["url", "publishedAt"])

        filter_cfg = self.configs.get("filter", {})
        self.exclude_title_keywords = [
            kw.lower() for kw in filter_cfg.get("exclude_title_keywords", [])
        ]
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.getenv(self.api_key_env)
            if not self._api_key:
                self.logger.error(f"Змінна оточення '{self.api_key_env}' не встановлена.")
        return self._api_key

    async def run(
        self,
        tickers: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Збирає новини з NewsAPI, фільтрує нові, зберігає в БД."""
        api_key = self._get_api_key()
        if not api_key:
            return None

        table_name = self.configs.get("table_name", "newsapi_articles")
        search_terms = list(set((tickers or []) + (keywords or [])))
        if not search_terms:
            self.logger.warning("[NewsAPI] Немає пошукових термінів. Пропускаємо.")
            return None

        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"terms": sorted(search_terms)}

        # 1. Кеш
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info("[NewsAPI] Cache hit — нових статей немає.")
                        return None
                    return new_from_cache

        # 2. Збір
        self.logger.info(f"[NewsAPI] Fetching for {len(search_terms)} terms...")
        tasks = [self._fetch_for_term(term, api_key) for term in search_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                all_articles.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(f"[NewsAPI] Error for '{search_terms[i]}': {res}")

        if not all_articles:
            self.logger.info("[NewsAPI] Статей не знайдено.")
            return None

        df = pd.DataFrame(all_articles)

        # 3. Hash
        df["hash"] = df.apply(
            lambda row: hashlib.sha256(
                "|".join(str(row.get(k, "")) for k in self.hash_keys).encode()
            ).hexdigest(),
            axis=1,
        )

        # 4. Кеш по хешу
        if self.cache_manager:
            is_new = df["hash"].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("[NewsAPI] Всі статті вже в кеші.")
                return None

        # 5. Фільтрація через БД
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[NewsAPI] Нових статей не знайдено в БД.")
            if self.cache_manager:
                for h in df["hash"]:
                    self.cache_manager.set(h, True, ttl=3600)
            return None

        # 6. Збереження
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            for h in new_df["hash"]:
                self.cache_manager.set(h, True, ttl=3600)
            self.cache_manager.set(
                cache_key, df.to_dict("records"), cache_params, namespace="collectors"
            )

        self.logger.info(f"[NewsAPI] Збережено {len(new_df)} нових статей.")
        return new_df

    async def _fetch_for_term(self, term: str, api_key: str) -> List[Dict[str, Any]]:
        params = {
            "q": f'"{term}"',
            "language": self.language,
            "pageSize": self.page_size,
            "apiKey": api_key,
        }
        try:
            client = self.http_client_factory.get_http_client()
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            filtered = []
            for a in articles:
                title = (a.get("title") or "").lower()
                if not any(kw in title for kw in self.exclude_title_keywords):
                    a["search_term"] = term
                    filtered.append(a)
            return filtered
        except Exception as e:
            self.logger.error(f"[NewsAPI] Помилка для '{term}': {e}")
            raise