import asyncio
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
    """A collector to fetch news articles from Google News, handling redirects."""
    collector_type = "google_news"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, cache_manager: Optional[CacheManager] = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.logger = ProjectLogger.get_logger(__name__)

        # Configure GNews instance with a date range and result limit
        period = self.configs.get('params', {}).get('period', '60d')
        max_results = self.configs.get('params', {}).get('max_results', 100)

        self.api = GNews(
            period=period,
            max_results=max_results
        )
        self.logger.info(f"GoogleNewsCollector configured for period '{period}' with max_results={max_results} per term.")

    async def run(self, tickers: Optional[List[str]] = None, keywords: Optional[List[str]] = None, **kwargs) -> Optional[pd.DataFrame]:
        """Fetches news, resolves redirects, avoids duplicates, and returns a DataFrame."""
        self.logger.info("Starting data collection for google_news...")
        
        search_terms = self._get_unique_search_terms(tickers, keywords)
        self.logger.info(f"Starting news collection from Google News for {len(search_terms)} unique terms.")

        all_articles = []
        client = self.http_client_factory.get_http_client()
        
        for term in search_terms:
            articles = await self._fetch_articles_for_term(term, client)
            all_articles.extend(articles)
            await asyncio.sleep(self.configs.get('delay', 1))

        if not all_articles:
            self.logger.info("No new articles found for any of the search terms.")
            return None

        self.logger.info(f"Collected {len(all_articles)} new raw articles from Google News.")
        
        raw_df = pd.DataFrame(all_articles)
        
        # Mark in cache after successful collection
        if self.cache_manager:
            for url in raw_df['link']:
                self.cache_manager.set(url, True)
            if 'original_link' in raw_df.columns:
                for url in raw_df['original_link']:
                    if url:
                         self.cache_manager.set(url, True)

        return raw_df

    def _get_unique_search_terms(self, tickers: Optional[List[str]], keywords: Optional[List[str]]) -> List[str]:
        search_terms = set()
        if tickers: search_terms.update(tickers)
        if keywords: search_terms.update(keywords)
        return list(search_terms)

    async def _resolve_redirect(self, url: str, client: httpx.AsyncClient) -> str:
        """Asynchronously resolves a redirect URL to its final destination."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = await client.head(url, follow_redirects=True, timeout=10, headers=headers)
            return str(response.url)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            self.logger.warning(f"Could not resolve redirect for {url}: {e}")
            return url

    async def _fetch_articles_for_term(self, term: str, client: httpx.AsyncClient) -> List[Dict]:
        """Asynchronously fetches articles for a term, resolving redirects."""
        try:
            self.logger.debug(f"Fetching feed for term '{term}'...")
            
            loop = asyncio.get_running_loop()
            news = await loop.run_in_executor(None, self.api.get_news, term)
            
            if not news:
                return []

            new_articles = []
            for entry in news:
                original_url = entry.get('url')
                if not original_url:
                    continue
                
                # Check cache
                if self.cache_manager and self.cache_manager.get(original_url) is not None:
                    continue

                final_url = await self._resolve_redirect(original_url, client)

                if self.cache_manager and self.cache_manager.get(final_url) is not None:
                    continue
                
                # Double check with DB if cache is not exhaustive
                if not self.cache_manager:
                    # (optional) we can do a DB check here but it's slower. 
                    # Stage will handle upsert which will prevent duplicates if DB is configured with unique keys.
                    pass

                processed_entry = self._process_entry(entry)
                if processed_entry:
                    processed_entry['link'] = final_url
                    processed_entry['original_link'] = original_url
                    new_articles.append(processed_entry)
            return new_articles
        except Exception as e:
            self.logger.error(f"Failed to fetch or process news for term '{term}': {e}", exc_info=True)
            return []

    def _process_entry(self, entry: Dict) -> Optional[Dict]:
        """Processes a single article entry from GNews."""
        if not entry.get('url'): return None
        return {
            "title": entry.get('title'),
            "link": entry.get('url'),
            "published_date": pd.to_datetime(entry.get('published date'), utc=True),
            "source": entry.get('publisher', {}).get('title'),
            "content": entry.get('description')
        }