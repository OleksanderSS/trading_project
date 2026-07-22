# src/data/collectors/sec_filings_collector.py

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class SECFilingsCollector(BaseCollector):
    """Collects current SEC filings via the EDGAR API protocol."""
    collector_type = "sec_filings"
    data_type = "fundamental"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,               # FIX: now explicitly defined in __init__
        cache_manager: CacheManager | None = None,
        config_manager: UnifiedConfigManager | None = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.config_manager = config_manager or kwargs.get("config_manager")
        self.submissions_url_template = self.configs.get("submissions_url_template")
        self.hash_keys = self.configs.get("hash_keys", ["accessionNumber", "cik"])

        if not self.submissions_url_template:
            raise ValueError("'submissions_url_template' must be specified in SEC config.")

        self._cik_map: dict[str, str] | None = None

    def _get_cik_map(self) -> dict[str, str]:
        if self._cik_map is None:
            try:
                assets_config = self.config_manager.get_config("assets")
                details = assets_config.get("details", {})
                self._cik_map = {
                    ticker: str(data["cik"])
                    for ticker, data in details.items()
                    if "cik" in data
                }
                logger.info(f"Loaded CIK map for {len(self._cik_map)} tickers.")
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f"Failed to load CIK map: {e}", exc_info=True)
                self._cik_map = {}
        return self._cik_map

    def _calculate_start_date(self, period: str, run_date: datetime) -> datetime:
        if "y" in period:
            days = int(period.replace("y", "")) * 365
        elif "d" in period:
            days = int(period.replace("d", ""))
        else:
            days = 60
        return run_date - timedelta(days=days)

    def _check_sec_cache(self, cache_key: str, cache_params: dict, table_name: str) -> pd.DataFrame | None:
        """Check cache for existing SEC filings data and filter new records."""
        if not self.cache_manager:
            return None
        cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
        if cached is not None:
            df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
            if "hash" in df_cached.columns:
                new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                if new_from_cache.empty:
                    logger.info("[SEC] Cache hit — no new filings detected.")
                    return None
                return new_from_cache
        return None

    def _get_valid_ciks(self, tickers: list[str]) -> dict[str, str] | None:
        """Get valid CIKs for provided tickers."""
        cik_map = self._get_cik_map()
        valid_ciks = {
            ticker: str(cik_map.get(ticker.upper(), "")).zfill(10)
            for ticker in tickers
            if ticker.upper() in cik_map
        }
        if not valid_ciks:
            logger.warning("No valid CIKs found for provided tickers.")
            return None
        return valid_ciks

    def _process_fetch_results(self, results: list, valid_ciks: dict[str, str]) -> list[dict[str, Any]]:
        """Process fetch results and extract filings."""
        all_filings: list[dict[str, Any]] = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                all_filings.extend(res)
            elif isinstance(res, Exception):
                ticker = list(valid_ciks.keys())[i]
                logger.exception(f"Error fetching filings for {ticker}: {res}")
        return all_filings

    def _create_filing_hash(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cryptographic hash for deduplication."""
        for key in self.hash_keys:
            if key in df.columns:
                df[key] = df[key].astype(str)
        df["hash"] = df.apply(
            lambda row: hashlib.sha256(
                "".join(str(row.get(k, "")) for k in self.hash_keys).encode()
            ).hexdigest(),
            axis=1,
        )
        return df

    def _update_sec_cache(self, cache_key: str, cache_params: dict, df: pd.DataFrame) -> None:
        """Update cache with SEC filings data."""
        if self.cache_manager:
            self.cache_manager.set(
                cache_key, df.to_dict("records"), cache_params, namespace="collectors"
            )

    async def run(self, tickers: list[str], **kwargs) -> pd.DataFrame | None:
        if not tickers:
            logger.warning("No tickers provided for SEC filings. Skipping.")
            return None

        table_name = self.configs.get("table_name", "sec_filings")
        run_date = kwargs.get("run_date", datetime.now())
        period_str = self.configs.get("params", {}).get("period", "60d")
        start_date = self._calculate_start_date(period_str, run_date)

        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {"tickers": sorted(tickers), "start_date": str(start_date.date())}

        # 1. Cache Verification
        cached_result = self._check_sec_cache(cache_key, cache_params, table_name)
        if cached_result is not None:
            return cached_result

        # 2. Sequential Data Acquisition
        valid_ciks = self._get_valid_ciks(tickers)
        if not valid_ciks:
            return None

        logger.info(f"[SEC] Fetching filings for {len(valid_ciks)} tickers from {start_date.date()}.")

        client = await self.http_client_factory.get_http_client()
        async with client:
            tasks = [
                self._fetch_filings_for_cik(ticker, cik, client, start_date)
                for ticker, cik in valid_ciks.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_filings = self._process_fetch_results(results, valid_ciks)

        if not all_filings:
            logger.info("[SEC] Zero raw filings retrieved from external queries.")
            return None

        df = pd.DataFrame(all_filings)

        # 3. Cryptographic Deduplication Hash
        df = self._create_filing_hash(df)

        # 4. Database Level Filtering
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            logger.info("[SEC] No novel filings identified against historical database.")
            self._update_sec_cache(cache_key, cache_params, df)
            return None

        # 5. Persistence to Storage
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])
        self._update_sec_cache(cache_key, cache_params, df)

        logger.info(f"[SEC] Successfully persisted {len(new_df)} new filings.")
        return new_df

    async def _fetch_filings_for_cik(
        self,
        ticker: str,
        cik: str,
        client: httpx.AsyncClient,
        start_date: datetime,
    ) -> list[dict[str, Any]]:
        url = self.submissions_url_template.format(cik=cik)
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            recent = data.get("filings", {}).get("recent", {})
            if not recent or "accessionNumber" not in recent:
                return []

            keys = list(recent.keys())
            count = len(recent["accessionNumber"])
            filings_list = [{k: recent[k][i] for k in keys} for i in range(count)]

            filtered = []
            for filing in filings_list:
                try:
                    filing_date = datetime.strptime(filing["filingDate"], "%Y-%m-%d")
                    if filing_date >= start_date:
                        filing["ticker"] = ticker
                        filing["cik"] = cik
                        # Serialize sub-arrays to JSON string equivalents
                        for k, v in filing.items():
                            if isinstance(v, list):
                                filing[k] = json.dumps(v)
                        filtered.append(filing)
                except (ValueError, TypeError):
                    continue

            return filtered

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error processing {ticker} (CIK: {cik}): {e}")
            raise RuntimeError(f"Failed to fetch SEC filings for {ticker} ({cik})") from e
