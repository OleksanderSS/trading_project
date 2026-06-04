# src/data/collectors/google_news_collector.py

import asyncio
import hashlib
from typing import Any

import pandas as pd
from gnews import GNews

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class GoogleNewsCollector(BaseCollector):
    """Collector for Google News API mapping streams. Excludes redirect resolving layers constraint mapping."""
    collector_type = "google_news"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.logger = ProjectLogger.get_logger(__name__)

        params = self.configs.get("params", {})
        period = params.get("period", "7d")          # Default 7d limit boundary constraints
        max_results = params.get("max_results", 10)   # Default 10 result boundary mapped payload

        self.api = GNews(period=period, max_results=max_results)
        # Set a per-request timeout on the underlying requests session used by gnews
        if hasattr(self.api, 'timeout'):
            self.api.timeout = 15  # 15s per individual HTTP request
        self.delay = self.configs.get("delay", 0.5)
        self.max_concurrent = self.configs.get("max_concurrent_terms", 5)

        # Quality parsing filter blocks
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
        tickers: list[str] | None = None,
        keywords: list[str] | None = None,
        search_terms: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Resolves target scopes mapping streams via database layer."""
        try:
            return await asyncio.wait_for(
                self._run_internal(tickers=tickers, keywords=keywords,
                                   search_terms=search_terms, **kwargs),
                timeout=120.0  # 2-minute hard cap for the entire collector
            )
        except TimeoutError:
            self.logger.warning(
                "[GoogleNews] Collector exceeded 120s total timeout. "
                "Returning None and continuing pipeline."
            )
            return None

    async def _run_internal(
        self,
        tickers: list[str] | None = None,
        keywords: list[str] | None = None,
        search_terms: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        # Normalize and concatenate keyword definitions scope parameters
        list(keywords.keys()) if isinstance(keywords, dict) else (keywords or [])
        # Use only tickers for news search to keep collection time bounded.
        # Generic keywords like 'fed', 'inflation' produce too many irrelevant results.
        all_terms = list(set(tickers or []))

        if not all_terms:
            self.logger.warning("GoogleNewsCollector: Search scope parameters are empty.")
            return None

        table_name = self.configs.get("table_name", "google_news")
        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"terms": sorted(all_terms)}

        # 1. State cache memory validation block constraints logic mapping
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info("[GoogleNews] Cache hit limit resolved — zero new articles identified in temporal scope boundary.")
                        return None
                    return new_from_cache

        self.logger.info(f"[GoogleNews] Initiating scope parameters constraints across {len(all_terms)} block limits (parallel stream blocks constraint index {self.max_concurrent})...")

        # 2. Block logical resolution using execution limits parameters index bindings
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._fetch_with_semaphore(term, semaphore) for term in all_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.error(f"Logic parse constraints execution mapped index block boundary failure '{all_terms[i]}': {res}")
            elif res:
                all_articles.extend(res)

        if not all_articles:
            self.logger.info("[GoogleNews] Zero novel entries mapped to execution layers.")
            return None

        # 3. Memory limit boundary limits mappings stream url definitions uniqueness check parameter binding boundaries.
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

        # 4. Hash identity comparison mapped resolution
        if self.cache_manager:
            is_new = df["hash"].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("[GoogleNews] Duplicative articles verified within local execution limits cache storage.")
                return None

        # 5. Filter layer constraints evaluation against historical boundary
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[GoogleNews] Historical layer identified zero execution constraint mapped articles matching memory constraint query limits protocol checks payload blocks matrix mappings scope parameter definition limits.")
            if self.cache_manager:
                for h in df["hash"]:
                    self.cache_manager.set(h, True, ttl=86400)
                self.cache_manager.set(cache_key, df.to_dict("records"), cache_params, namespace="collectors")
            return None

        # 6. Database save mapped constraint bounds
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            for h in new_df["hash"]:
                self.cache_manager.set(h, True, ttl=86400)
            self.cache_manager.set(cache_key, df.to_dict("records"), cache_params, namespace="collectors")

        self.logger.info(f"[GoogleNews] Recorded bound {len(new_df)} articles limits check constraint boundary.")
        return new_df

    async def _fetch_with_semaphore(self, term: str, semaphore: asyncio.Semaphore) -> list[dict]:
        async with semaphore:
            result = await self._fetch_articles_for_term(term)
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            return result

    async def _fetch_articles_for_term(self, term: str) -> list[dict]:
        """Resolves logic mapped API constraint boundaries directly ignoring recursive limit blocks structure logic index mappings redirect scopes checks parameter constraint execution limits"""
        try:
            loop = asyncio.get_running_loop()
            # Wrap in asyncio.wait_for to prevent indefinite hangs on slow/blocked requests
            news = await asyncio.wait_for(
                loop.run_in_executor(None, self.api.get_news, term),
                timeout=30.0
            )
            if not news:
                return []

            articles = []
            for entry in news:
                processed = self._process_entry(entry)
                if processed:
                    articles.append(processed)
            return articles

        except TimeoutError:
            self.logger.warning(f"[GoogleNews] Timeout fetching news for '{term}' (30s). Skipping.")
            return []
        except Exception as e:
            self.logger.error(f"Resolution failed mapping boundary limits: '{term}': {e}")
            raise RuntimeError(f"Failed to fetch Google News articles for {term}") from e

    def _process_entry(self, entry: dict) -> dict | None:
        url = entry.get("url")
        if not url:
            return None

        title = entry.get("title", "") or ""
        source = entry.get("publisher", {}).get("title", "") or ""

        # Filter boundaries structure blocks mapping limits text
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
