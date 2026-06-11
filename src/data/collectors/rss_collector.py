import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser
import httpx
import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 15.0
_PARSE_TIMEOUT = 10.0
_FEED_TIMEOUT = 30.0
_TOTAL_TIMEOUT = 120.0


class RSSCollector(BaseCollector):
    """
    Collector for news from RSS feeds.

    Every async boundary has an explicit timeout so a single dead server
    can never stall the whole pipeline:

        asyncio.gather(tasks)           ← return_exceptions=True
            └─ asyncio.wait_for(feed, _FEED_TIMEOUT)
                    ├─ httpx.get()      ← _CONNECT_TIMEOUT / _READ_TIMEOUT
                    └─ feedparser       ← _PARSE_TIMEOUT  (executor)

    The entire collector.run() is wrapped in _TOTAL_TIMEOUT by Stage-1.
    """
    collector_type = 'rss'
    data_type = 'news'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs) ->None:
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)
        self.logger = ProjectLogger.get_logger(__name__)
        # ✅ Store config_manager so _get_rss_feeds can load feeds from knowledge_base
        self.config_manager = kwargs.get('config_manager')
        self.period_days: int = self._parse_period_to_days()
        filter_cfg = self.configs.get('filter', {})
        self.min_source_quality: float = filter_cfg.get('min_source_quality',
            0.0)
        self.exclude_title_keywords: list = [kw.lower() for kw in
            filter_cfg.get('exclude_title_keywords', [])]
        self._quality_weights: dict[str, float] = kwargs.get('quality_weights',
            {})
        self._semaphore = asyncio.Semaphore(5)
        self.logger.info(
            f'RSSCollector initialised | period={self.period_days}d min_quality={self.min_source_quality} concurrency=5'
            )

    async def run(self, tickers: list[str] | None=None, keywords:
        list[str] | None=None, **kwargs) ->pd.DataFrame | None:
        return await self._run_internal(tickers=tickers, keywords=keywords,
            **kwargs)

    def _parse_period_to_days(self) ->int:
        period_str = self.configs.get('params', {}).get('period', '7d')
        return int(period_str.replace('d', '')) if 'd' in period_str else 7

    def _check_rss_cache(self, cache_key: str, cache_params: dict, table_name: str) -> pd.DataFrame | None:
        """Check cache for existing RSS data and filter new records."""
        if not self.cache_manager:
            return None
        cached = self.cache_manager.get(cache_key, cache_params, namespace='collectors')
        if cached is not None:
            df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
            if 'hash' in df_cached.columns:
                new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                if new_from_cache.empty:
                    self.logger.info('[RSS] Cache hit – no new articles.')
                    return None
                return new_from_cache
        return None

    def _get_rss_feeds(self, **kwargs) -> list[dict] | None:
        """Get RSS feeds from configuration or kwargs."""
        feeds = kwargs.get('rss_feeds') or self.configs.get('feeds', [])
        if not feeds:
            config_manager = kwargs.get('config_manager') or getattr(self, 'config_manager', None)
            if config_manager:
                kb = config_manager.get_config('knowledge_base')
                feeds = (kb or {}).get('rss_feeds', [])
        if not feeds:
            self.logger.warning('[RSS] No feeds configured. Skipping.')
            return None
        return feeds

    def _process_feed_results(self, results: list, feeds: list[dict]) -> list[dict]:
        """Process feed fetch results and extract articles."""
        flat_articles: list[dict] = []
        for i, res in enumerate(results):
            feed_name = feeds[i]['name'] if i < len(feeds) else 'unknown'
            if isinstance(res, asyncio.TimeoutError):
                self.logger.warning(f"[RSS] Feed '{feed_name}' timed out after {_FEED_TIMEOUT}s – skipped.")
            elif isinstance(res, Exception):
                self.logger.error(f"[RSS] Feed '{feed_name}' raised {type(res).__name__}: {res}")
            elif isinstance(res, list):
                flat_articles.extend(res)
        return flat_articles

    def _filter_rss_articles(self, df: pd.DataFrame, table_name: str, cache_key: str, cache_params: dict) -> pd.DataFrame | None:
        """Filter RSS articles by cache and database."""
        df['hash'] = df['link'].apply(lambda url: hashlib.sha256(str(url).encode()).hexdigest())

        if self.cache_manager:
            df = df[df['hash'].apply(lambda h: self.cache_manager.get(h) is None)].copy()
            if df.empty:
                self.logger.info('[RSS] All articles already in active cache.')
                return None

        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info('[RSS] No novel articles vs. historical DB.')
            self._update_cache(cache_key, cache_params, df)
            return None

        return new_df

    async def _run_internal(self, tickers: list[str] | None=None,
        keywords: list[str] | None=None, **kwargs) ->pd.DataFrame | None:
        table_name = self.configs.get('table_name', 'rss_news')
        cache_key = f'{self.__class__.__name__}_run'
        cache_params = {'period_days': self.period_days}

        # Check cache first
        cached_result = self._check_rss_cache(cache_key, cache_params, table_name)
        if cached_result is not None:
            return cached_result

        # Get RSS feeds
        feeds = self._get_rss_feeds(**kwargs)
        if not feeds:
            return None

        # Fetch feeds
        self.logger.info(f'[RSS] Fetching {len(feeds)} feeds (semaphore=5, feed_timeout={_FEED_TIMEOUT}s)…')
        tasks = [asyncio.wait_for(self._fetch_feed(feed['name'], feed['url']), timeout=_FEED_TIMEOUT) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        flat_articles = self._process_feed_results(results, feeds)
        if not flat_articles:
            self.logger.info('[RSS] No new articles from any feed.')
            return None

        self.logger.info(f'[RSS] Collected {len(flat_articles)} raw articles total.')
        df = pd.DataFrame(flat_articles)

        # Filter articles
        new_df = self._filter_rss_articles(df, table_name, cache_key, cache_params)
        if new_df is None:
            return None

        # Persist new articles
        self.db_manager.upsert(table_name, new_df, unique_on=['hash'])
        self._update_cache(cache_key, cache_params, df, new_df)
        self.logger.info(f'[RSS] Persisted {len(new_df)} new articles.')
        return new_df

    def _update_cache(self, cache_key: str, cache_params: dict, df: pd.
        DataFrame, new_df: pd.DataFrame | None=None) ->None:
        if not self.cache_manager:
            return
        hashes = new_df['hash'] if new_df is not None else df['hash']
        for h in hashes:
            self.cache_manager.set(h, True, ttl=86400)
        self.cache_manager.set(cache_key, df.to_dict('records'),
            cache_params, namespace='collectors')

    async def _fetch_feed(self, name: str, url: str) ->list[dict]:
        """
        Fetch + parse one RSS feed.

        Both the HTTP request and the feedparser call have independent
        timeouts so neither can block the event loop indefinitely.
        """
        http_timeout = httpx.Timeout(timeout=_READ_TIMEOUT, connect=
            _CONNECT_TIMEOUT, read=_READ_TIMEOUT, pool=_CONNECT_TIMEOUT)
        async with self._semaphore:
            self.logger.info(f"[RSS] → fetching '{name}'")
            try:
                async with httpx.AsyncClient(timeout=http_timeout,
                    follow_redirects=True) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    response_text = response.text
            except httpx.TimeoutException as exc:
                self.logger.warning(
                    f"[RSS] '{name}' HTTP timeout ({type(exc).__name__}). Skipping."
                    )
                return []
            except Exception as exc:
                self.logger.error(f'Виникла помилка: {exc}', exc_info=True)
                self.logger.warning(
                    f"[RSS] '{name}' HTTP error ({type(exc).__name__}): {exc}. Skipping."
                    )
                raise RuntimeError(f"RSS feed '{name}' HTTP error") from exc
            try:
                loop = asyncio.get_running_loop()
                feed_data = await asyncio.wait_for(loop.run_in_executor(
                    None, lambda : feedparser.parse(response_text)),
                    timeout=_PARSE_TIMEOUT)
            except TimeoutError:
                self.logger.warning(
                    f"[RSS] '{name}' feedparser timed out after {_PARSE_TIMEOUT}s. Skipping."
                    )
                return []
            except Exception as exc:
                self.logger.error(f"[RSS] '{name}' parse error: {exc}",
                    exc_info=True)
                raise RuntimeError(f"RSS feed '{name}' parse error") from exc
            limit = self.configs.get('params', {}).get('limit_per_feed', 20)
            cutoff = datetime.now(UTC) - timedelta(days=self.
                period_days)
            entries = feed_data.entries[:limit]
            self.logger.info(
                f"[RSS] '{name}' retrieved – parsing {len(entries)} entries.")
            articles: list[dict] = []
            skipped_old = skipped_filter = 0
            for entry in entries:
                processed = self._process_entry(entry, name)
                if processed is None:
                    skipped_filter += 1
                    continue
                if processed['published_date'] < cutoff:
                    skipped_old += 1
                    continue
                articles.append(processed)
            self.logger.info(
                f"[RSS] '{name}' done: kept={len(articles)} skipped_filter={skipped_filter} skipped_old={skipped_old}"
                )
            return articles

    def _process_entry(self, entry: Any, source_name: str) ->dict | None:
        published_str = entry.get('published')
        if not published_str:
            return None
        try:
            published_date = pd.to_datetime(published_str, utc=True)
        except (ValueError, TypeError):
            return None
        title = entry.get('title', '') or ''
        if any(kw in title.lower() for kw in self.exclude_title_keywords):
            return None
        if self.min_source_quality > 0 and self._quality_weights:
            quality = self._quality_weights.get(source_name.lower(), self.
                _quality_weights.get('default_weight', 0.3))
            if quality < self.min_source_quality:
                return None
        return {'title': title, 'link': entry.get('link'), 'published_date':
            published_date, 'source': source_name, 'content': entry.get(
            'summary')}
