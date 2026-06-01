# src/data/collectors/insider_collector.py

import asyncio
import httpx
import logging
import pandas as pd
import hashlib
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class InsiderCollector(BaseCollector):
    """
    Asynchronously aggregates insider trading execution reports via OpenInsider scraping.
    """
    collector_type = "insider"
    data_type = "alternative"

    def __init__(
        self,
        configs: Dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: Optional[CacheManager] = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.hash_keys = self.configs.get("hash_keys", ["filing_date", "ticker", "insider_name"])

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Asynchronously parses execution lists from HTTP sources passed through configuration payload.
        """
        urls_to_scrape = self.configs.get("urls")
        if not urls_to_scrape:
            self.logger.warning(f"No execution URLs specified in '{self.collector_type}' conf. Skipping execution.")
            return []

        self.logger.info(f"Initiating asynchronous HTTP polling across {len(urls_to_scrape)} URIs.")
        all_trades: List[Dict[str, Any]] = []
        
        # Get async HTTP client from factory (don't use context manager if not supported)
        try:
            client = await self.http_client_factory.get_http_client()
            # Check if it's async-capable (has httpx client)
            if hasattr(client, 'get') and asyncio.iscoroutinefunction(getattr(client, 'get')):
                tasks = [self._scrape_url(url, client) for url in urls_to_scrape]
            else:
                # Fallback: wrap blocking client in thread pool
                tasks = [asyncio.to_thread(self._scrape_url_sync, url) for url in urls_to_scrape]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Network error parsing URL resource {urls_to_scrape[i]}: {result}")
                elif result:
                    all_trades.extend(result)
                
                # Rate limiting: delay between requests
                await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Network interface initialization failed: {e}")
            return []

        self.logger.info(f"Total isolated records aggregated: {len(all_trades)}.")
        return all_trades

    async def _scrape_url(self, url: str, client: httpx.AsyncClient) -> Optional[List[Dict[str, Any]]]:
        """
        Processes an isolated asynchronous HTTP network request resolving targeted DOM nodes.
        """
        user_agent = self.configs.get("user_agent", "Mozilla/5.0")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Parsing DOM layout from URL: {url}")
        headers = {"User-Agent": user_agent}
        
        try:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            return self._parse_html(response.text, url)

        except Exception as e:
            self.logger.error(f"Error handling DOM layout state from {url}: {e}")
            raise e

    def _scrape_url_sync(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Synchronous network query resolution protocol constraint mapping.
        """
        user_agent = self.configs.get("user_agent", "Mozilla/5.0")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Synchronous DOM parse query at url: {url}")
        headers = {"User-Agent": user_agent}
        
        try:
            # Use httpx synchronous client for fallback
            import httpx as sync_httpx
            with sync_httpx.Client(timeout=10) as sync_client:
                response = sync_client.get(url, headers=headers)
                response.raise_for_status()
                return self._parse_html(response.text, url)
        except Exception as e:
            self.logger.error(f"Synchronous execution block error {url}: {e}")
            raise e

    def _parse_html(self, html: str, url: str) -> List[Dict[str, Any]]:
        """Parses logical HTML nodes rendering matrix representations string outputs."""
        column_mapping = self.configs.get("column_mapping", {})
        if not column_mapping:
            raise ValueError(f"Mapping configurations missing in execution node '{self.collector_name}'.")

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="tinytable")
        
        if not table:
            self.logger.warning(f"Failed to identify primary extraction table class bounds under URL: {url}.")
            return []

        rows = table.find_all("tr")[1:]  # Pass headers constraint logic block
        parsed_trades = []
        expected_col_count = len(column_mapping)

        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < expected_col_count:
                continue

            # Project mapping logical columns into node definitions
            trade_data = {}
            for key, field_name in column_mapping.items():
                if key.startswith("col_"):
                    try:
                        col_idx = int(key.split("_")[1])
                        if col_idx < len(cells):
                            trade_data[field_name] = cells[col_idx]
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping invalid column index {key}: {e}")
            
            if trade_data:
                parsed_trades.append(trade_data)

        return parsed_trades

    async def run(
        self,
        tickers: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetches and deduplicates targeted logical execution trades data resolving historical parameters."""
        table_name = self.configs.get("table_name", "insider_trades")
        
        # Abort if logic bounds lack URIs definitions
        urls = self.configs.get("urls")
        if not urls:
            self.logger.warning("[Insider] Aborted runtime context: Configuration constraint limits URL scope definition.")
            return None

        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"urls": sorted(urls) if isinstance(urls, list) else [urls]}

        # 1. Cache Verification
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info("[Insider] Logical boundary limits active record propagation. Cache hit verified.")
                        return None
                    return new_from_cache

        # 2. Data Acquisition Target Resolution
        self.logger.info("[Insider] Fetching insider trades...")
        try:
            raw_data = await self.fetch_raw_data(**kwargs)
        except Exception as e:
            self.logger.error(f"[Insider] Contextual data parse process aborted: {e}")
            return None

        if not raw_data:
            self.logger.info("[Insider] No historical records extracted.")
            return None

        df = pd.DataFrame(raw_data)

        # 3. Hash Validation Algorithm Definition
        df["hash"] = df.apply(
            lambda row: hashlib.sha256(
                "|".join(str(row.get(k, "")) for k in self.hash_keys).encode()
            ).hexdigest(),
            axis=1,
        )

        # 4. Filter validation constraint layer
        if self.cache_manager:
            is_new = df["hash"].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("[Insider] Primary logical index boundary satisfied in memory store.")
                return None

        # 5. Database logic mapping integration
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[Insider] Verification check identified duplicate historical matrix representations.")
            if self.cache_manager:
                for h in df["hash"]:
                    self.cache_manager.set(h, True, ttl=86400)
            return None

        # 6. Saving integration parameters bounds
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            for h in new_df["hash"]:
                self.cache_manager.set(h, True, ttl=86400)
            self.cache_manager.set(
                cache_key, df.to_dict("records"), cache_params, namespace="collectors"
            )

        self.logger.info(f"[Insider] Committed bounded record list limits constraint size of {len(new_df)}.")
        return new_df

