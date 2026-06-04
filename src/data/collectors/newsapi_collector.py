# src/data/collectors/newsapi_collector.py

import asyncio
import hashlib
import os
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class NewsAPICollector(BaseCollector):
    """Collector for fetching news streams from NewsAPI endpoints."""

    collector_type = "newsapi"
    data_type = "news"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        **kwargs,
    ):
        super().__init__(
            configs, http_client_factory, db_manager, cache_manager, **kwargs
        )
        self.base_url = self.configs.get(
            "base_url", "https://newsapi.org/v2/everything"
        )
        self.language = self.configs.get("language", "en")
        self.page_size = self.configs.get("page_size", 20)
        self.hash_keys = self.configs.get("hash_keys", ["url", "publishedAt"])

        filter_cfg = self.configs.get("filter", {})
        self.exclude_title_keywords = [
            kw.lower() for kw in filter_cfg.get("exclude_title_keywords", [])
        ]
        # api_key_name contains the env var name (e.g. "NEWS_API_KEY"), resolve it
        api_key_var = self.configs.get("api_key_name", "NEWS_API_KEY")
        self._api_key: str | None = os.getenv(api_key_var)

    def _get_api_key(self) -> str | None:
        if self._api_key is None:
            self.logger.error(
                "[NewsAPI] No API key available."
            )
        return self._api_key

    async def run(
        self,
        tickers: list[str] | None = None,
        keywords: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Fetch news from NewsAPI, filter novel records, commit to DB."""
        api_key = self._get_api_key()
        if not api_key:
            return None

        table_name = self.configs.get("table_name", "newsapi_articles")
        search_terms = list(set((tickers or []) + (keywords or [])))
        if not search_terms:
            self.logger.warning(
                "[NewsAPI] No search terms provided. Skipping execution."
            )
            return None

        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"terms": sorted(search_terms)}

        # 1. State Verification (Cache lookup)
        if self.cache_manager:
            cached = self.cache_manager.get(
                cache_key, cache_params, namespace="collectors"
            )
            if cached is not None:
                df_cached = (
                    pd.DataFrame(cached)
                    if isinstance(cached, list)
                    else cached
                )
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(
                        table_name, df_cached
                    )
                    if new_from_cache.empty:
                        self.logger.info(
                            "[NewsAPI] Cache hit — no new articles detected."
                        )
                        return None
                    return new_from_cache

        # 2. Sequential Data Acquisition
        self.logger.info(
            f"[NewsAPI] Issuing collection requests for {len(search_terms)} terms..."
        )
        tasks = [self._fetch_for_term(term, api_key) for term in search_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                all_articles.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(
                    f"[NewsAPI] Network error for term '{search_terms[i]}': {res}"
                )

        if not all_articles:
            self.logger.info(
                "[NewsAPI] Zero articles retrieved from external queries."
            )
            return None

        df = pd.DataFrame(all_articles)

        # 3. Cryptographic Deduplication Hash
        df["hash"] = df.apply(
            lambda row: hashlib.sha256(
                "|".join(str(row.get(k, "")) for k in self.hash_keys).encode()
            ).hexdigest(),
            axis=1,
        )

        # 4. Filter against Hash Memory
        if self.cache_manager:
            is_new = df["hash"].apply(
                lambda h: self.cache_manager.get(h) is None
            )
            df = df[is_new].copy()
            if df.empty:
                self.logger.info(
                    "[NewsAPI] All fetched articles already exist in active cache."
                )
                return None

        # 5. Filter against Historical Database
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info(
                "[NewsAPI] No novel articles identified against historical database."
            )
            if self.cache_manager:
                for h in df["hash"]:
                    self.cache_manager.set(h, True, ttl=3600)
            return None

        # 6. Persistence to Storage
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            for h in new_df["hash"]:
                self.cache_manager.set(h, True, ttl=3600)
            self.cache_manager.set(
                cache_key,
                df.to_dict("records"),
                cache_params,
                namespace="collectors",
            )

        self.logger.info(
            f"[NewsAPI] Successfully persisted {len(new_df)} new articles."
        )
        return new_df

    async def _fetch_for_term(
        self, term: str, api_key: str
    ) -> list[dict[str, Any]]:
        params = {
            "q": f'"{term}"',
            "language": self.language,
            "pageSize": self.page_size,
            "apiKey": api_key,
        }
        try:
            client = await self.http_client_factory.get_http_client()
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
            self.logger.error(
                f"[NewsAPI] HTTP context error for '{term}': {e}"
            )
            raise

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict("records") if df is not None else None

