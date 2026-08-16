# src/data/collectors/insider_collector.py

import asyncio
import hashlib
import logging
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class InsiderCollector(BaseCollector):
    """
    Asynchronously aggregates insider trading execution reports via OpenInsider scraping.
    """
    collector_type = "insider"
    data_type = "alternative"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.hash_keys = self.configs.get("hash_keys", ["filing_date", "ticker", "insider_name"])

    @staticmethod
    def _urls_for_tickers(templates: list[str], tickers: list[str] | None) -> list[str]:
        """One query per ticker, built from the configured URL as a template.

        The configured screener URL carries `s=` empty -- no symbol filter --
        with `cnt=100&page=1`, so every run fetched the hundred most recent
        filings ACROSS THE WHOLE MARKET. `tickers` was accepted by the caller
        and never used.

        The consequence is not an error, it is a coverage ceiling, and it is
        why the feature looked broken twice. Of 1,395 accumulated filings just
        11 concern our 22 companies, and `insider_net_value_30d` reaches 5% of
        bars. Nothing in the pipeline could have fixed that downstream: the
        rows were never collected.

        The query shape stays in the config -- only `s` is substituted -- so
        changing the window or the row count remains a config edit.
        """
        if not tickers:
            return list(templates)

        wanted = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
        if not wanted:
            return list(templates)

        urls: list[str] = []
        for template in templates:
            split = urlsplit(str(template))
            params = parse_qsl(split.query, keep_blank_values=True)
            for symbol in wanted:
                rebuilt = [(k, symbol if k == "s" else v) for k, v in params]
                if not any(k == "s" for k, _ in rebuilt):
                    rebuilt.append(("s", symbol))
                urls.append(urlunsplit(split._replace(query=urlencode(rebuilt))))
        return urls

    async def fetch_raw_data(self, tickers: list[str] | None = None, **kwargs) -> list[dict[str, Any]]:
        """
        Asynchronously parses execution lists from HTTP sources passed through configuration payload.
        """
        urls_to_scrape = self.configs.get("urls")
        if not urls_to_scrape:
            self.logger.warning(f"No execution URLs specified in '{self.collector_type}' conf. Skipping execution.")
            return []

        urls_to_scrape = self._urls_for_tickers(urls_to_scrape, tickers)
        self.logger.info(f"Initiating asynchronous HTTP polling across {len(urls_to_scrape)} URIs.")
        all_trades: list[dict[str, Any]] = []

        # Get async HTTP client from factory (don't use context manager if not supported)
        try:
            client = await self.http_client_factory.get_http_client()

            # One URL became twenty-two when the query gained a symbol filter,
            # and `gather` would have fired all of them at one small free site
            # simultaneously. A few at a time costs seconds and is the
            # difference between using a source and abusing it.
            gate = asyncio.Semaphore(int(self.configs.get("max_concurrent_requests", 4)))

            async def _limited(coro_factory):
                async with gate:
                    return await coro_factory()

            # Check if it's async-capable (has httpx client)
            if hasattr(client, 'get') and asyncio.iscoroutinefunction(client.get):
                tasks = [
                    _limited(lambda u=url: self._scrape_url(u, client))
                    for url in urls_to_scrape
                ]
            else:
                # Fallback: wrap blocking client in thread pool
                tasks = [
                    _limited(lambda u=url: asyncio.to_thread(self._scrape_url_sync, u))
                    for url in urls_to_scrape
                ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.exception(f"Network error parsing URL resource {urls_to_scrape[i]}: {result}")
                elif result:
                    all_trades.extend(result)

                # Rate limiting: delay between requests
                await asyncio.sleep(1)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Network interface initialization failed: {e}")
            raise RuntimeError("Failed to initialize insider collector network interface") from e

        self.logger.info(f"Total isolated records aggregated: {len(all_trades)}.")
        return all_trades

    async def _scrape_url(self, url: str, client: httpx.AsyncClient) -> list[dict[str, Any]] | None:
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

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error handling DOM layout state from {url}: {e}")
            raise e

    def _scrape_url_sync(self, url: str) -> list[dict[str, Any]] | None:
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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Synchronous execution block error {url}: {e}")
            raise e

    def _parse_html(self, html: str, url: str) -> list[dict[str, Any]]:
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
        index_by_field = self._normalized_column_mapping(column_mapping)
        if not index_by_field:
            self.logger.error(
                f"column_mapping for '{self.collector_type}' has no usable "
                f"field->index entries: {column_mapping!r}"
            )
            return []

        # Every mapped index must actually exist in the row.
        required_width = max(index_by_field.values()) + 1

        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < required_width:
                continue

            trade_data = {
                field_name: cells[col_idx]
                for field_name, col_idx in index_by_field.items()
            }
            if trade_data:
                parsed_trades.append(trade_data)

        if not parsed_trades and rows:
            self.logger.warning(
                f"Found {len(rows)} table rows at {url} but mapped zero trades "
                f"(need >= {required_width} cells per row). column_mapping is "
                f"probably out of sync with the page layout."
            )

        return parsed_trades

    @staticmethod
    def _normalized_column_mapping(column_mapping: dict[str, Any]) -> dict[str, int]:
        """Normalize `column_mapping` to {field_name: column_index}.

        Two shapes exist in the wild and the parser used to accept only the
        one nobody actually configured:

        - `{field_name: index}` -- what `src/config/collectors.yaml` really
          uses (`filing_date: 0`, `ticker: 2`, ...). The old code tested
          `key.startswith("col_")` against these keys, which is never true,
          so `trade_data` came out empty for every single row and the
          collector silently returned zero trades on a perfectly good page.
        - `{"col_<index>": field_name}` -- the shape the old code expected.
          Still accepted so any config written that way keeps working.
        """
        normalized: dict[str, int] = {}
        for key, value in column_mapping.items():
            if isinstance(key, str) and key.startswith("col_"):
                try:
                    normalized[str(value)] = int(key.split("_", 1)[1])
                except (ValueError, IndexError):
                    logger.debug(f"Skipping invalid column index key {key!r}")
                continue
            try:
                normalized[str(key)] = int(value)
            except (TypeError, ValueError):
                logger.debug(f"Skipping non-integer column index for {key!r}: {value!r}")
        return normalized

    def _check_cache(self, cache_key: str, cache_params: dict, table_name: str) -> pd.DataFrame | None:
        """Check cache for existing data and filter new records."""
        if not self.cache_manager:
            return None
        cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
        if cached is not None:
            df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
            if "hash" in df_cached.columns:
                new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                if new_from_cache.empty:
                    self.logger.info("[Insider] Logical boundary limits active record propagation. Cache hit verified.")
                    return None
                return new_from_cache
        return None

    def _update_cache(self, df: pd.DataFrame) -> None:
        """No-op retained for call-site compatibility.

        This used to write one CacheManager entry per trade hash
        (`set(h, True, ttl=86400)`), paired with a per-row `get(h)` in a
        now-removed `_filter_by_cache_manager`. Each `set` is a pickle write
        plus a single-row DuckDB upsert into `cache_metadata`; each `get` is
        its own `SELECT ... WHERE key_hash = ?`. `filter_new_records()`
        already dedups on the same `hash` column in ONE query and
        `upsert(unique_on=['hash'])` enforces it at write time, so the
        markers were duplicated work with no added safety.
        """
        return

    async def run(
        self,
        tickers: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Fetches and deduplicates targeted logical execution trades data resolving historical parameters."""
        table_name = self.configs.get("table_name", "insider_trades")

        # Abort if logic bounds lack URIs definitions
        urls = self.configs.get("urls")
        if not urls:
            self.logger.warning("[Insider] Aborted runtime context: Configuration constraint limits URL scope definition.")
            return None

        cache_key = f"{self.__class__.__name__}_run"
        # The ticker set belongs in the key because it now decides which URLs
        # are fetched. Keyed on the configured URLs alone, adding a company to
        # the config would have been served the previous company's cache.
        cache_params = {
            "urls": sorted(urls) if isinstance(urls, list) else [urls],
            "tickers": sorted({str(t).strip().upper() for t in (tickers or []) if str(t).strip()}),
        }

        # 1. Cache Verification
        cached_result = self._check_cache(cache_key, cache_params, table_name)
        if cached_result is not None:
            return cached_result

        # 2. Data Acquisition Target Resolution
        self.logger.info("[Insider] Fetching insider trades...")
        try:
            # `tickers` is a named parameter of `run`, so it is NOT in
            # `**kwargs`, and this line dropped it one step before the only
            # place that uses it. The orchestrator passes the list, the
            # screener query kept `s=` empty, and the run fetched the hundred
            # most recent filings across the whole market -- with the log
            # saying "polling across 1 URIs" instead of 22.
            raw_data = await self.fetch_raw_data(tickers=tickers, **kwargs)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"[Insider] Contextual data parse process aborted: {e}")
            raise RuntimeError("Insider collection failed") from e

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

        # 4. Database logic mapping integration
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[Insider] Verification check identified duplicate historical matrix representations.")
            self._update_cache(df)
            return None

        # 6. Saving integration parameters bounds
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])
        self._update_cache(new_df)

        self.logger.info(f"[Insider] Committed bounded record list limits constraint size of {len(new_df)}.")
        return new_df

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None

