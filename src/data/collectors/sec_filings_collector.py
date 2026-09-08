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
                if isinstance(res, httpx.HTTPStatusError) and res.response.status_code == 404:
                    # A data fact, not a failure: SEC has no submissions file
                    # for that number. IWM sat on 0001112953, which is not in
                    # SEC's own ticker map at all.
                    logger.warning(
                        "[SEC] No submissions for %s at CIK%s. The number in "
                        "assets.yaml does not identify a filer -- check it "
                        "against https://www.sec.gov/files/company_tickers.json",
                        ticker, valid_ciks[ticker],
                    )
                else:
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

        # The SEC has a strict limit of 10 requests per second.
        # Use a semaphore to limit concurrency and avoid ConnectTimeout / drops.
        sem = asyncio.Semaphore(5)
        
        async def fetch_with_sem(ticker, cik):
            async with sem:
                # Add a small delay to further ensure we don't burst past the 10/sec limit
                await asyncio.sleep(0.2)
                return await self._fetch_filings_for_cik(ticker, cik, client, start_date)

        client = await self.http_client_factory.get_http_client()
        async with client:
            tasks = [
                fetch_with_sem(ticker, cik)
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

    @staticmethod
    def _block_to_filings(
        block: dict[str, Any],
        ticker: str,
        cik: str,
        start_date: datetime,
    ) -> list[dict[str, Any]]:
        """One filings block -> rows, filtered to the window.

        `filings.recent` and each file under `filings.files` have the SAME
        shape -- parallel arrays keyed by field name, the older batch being
        that block as the whole document. So they share this, rather than the
        older path getting a second copy of the date filter and the list
        serialisation to drift away from.
        """
        if not block or "accessionNumber" not in block:
            return []
        keys = list(block.keys())
        rows: list[dict[str, Any]] = []
        for index in range(len(block["accessionNumber"])):
            filing = {key: block[key][index] for key in keys}
            try:
                filed = datetime.strptime(filing["filingDate"], "%Y-%m-%d")
            except (ValueError, TypeError, KeyError):
                continue
            if filed < start_date:
                continue
            filing["ticker"] = ticker
            filing["cik"] = cik
            # Serialize sub-arrays to JSON string equivalents
            for key, value in filing.items():
                if isinstance(value, list):
                    filing[key] = json.dumps(value)
            rows.append(filing)
        return rows

    async def _fetch_filings_for_cik(
        self,
        ticker: str,
        cik: str,
        client: httpx.AsyncClient,
        start_date: datetime,
    ) -> list[dict[str, Any]]:
        url = self.submissions_url_template.format(cik=cik)
        # SEC EDGAR requires a specific User-Agent format: 'CompanyName ContactEmail'
        headers = {"User-Agent": "DEAN_OS_Agent research@example.com", "Accept-Encoding": "gzip, deflate"}

        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Say whose filings these are. A CIK that resolves is not a CIK
            # that is right: SPY was configured as 0000896976, which returns
            # HTTP 200 for 'VAN KAMPEN AMERICAN CAPITAL EQUITY OPPORTUNITY
            # TRUST SER 14' -- 24 filings, all between 1995 and 2001. Nothing
            # fell inside the collection window, so the collector reported no
            # error and contributed nothing, silently, on every run. Printing
            # the entity makes the next such mismatch visible in the log
            # instead of requiring an audit against SEC to find it.
            entity = data.get("name")
            if entity:
                logger.info("[SEC] %s -> CIK%s %r", ticker, cik, entity)

            recent = data.get("filings", {}).get("recent", {})
            if not recent or "accessionNumber" not in recent:
                return []

            filtered = self._block_to_filings(recent, ticker, cik, start_date)

            # The older batches, which this collector did not read until
            # 2026-09-08. `recent` is capped at 1000 filings, so how far back
            # it reaches is decided by how OFTEN a company files, and the
            # result was a coverage curve that made the data unusable for
            # measurement rather than merely thin:
            #
            #   1997-2012   11 to 157 filings a year, from 2 to 6 tickers
            #   2019-2023   5,894 to 9,036 a year, from 70 to 94 tickers
            #
            # A filing-based feature measured over the explorable period would
            # therefore be a statement about the last five years and about the
            # names that file most, wearing the label of a thirty-year result.
            #
            # `filings.files` carries the rest, free, in the same shape: for
            # AAPL one batch of 1,246 reaching 1994-01-26, for KO and XOM two
            # each reaching 1994 and holding 2,301 and 2,554. It is the same
            # host and the same parser; only the URL differs.
            for batch in data.get("filings", {}).get("files", []) or []:
                name = batch.get("name")
                if not name:
                    continue
                # Skip a batch that ends before the window opens: its URL costs
                # a request and its contents are all discarded.
                if batch.get("filingTo") and batch["filingTo"] < start_date.strftime("%Y-%m-%d"):
                    continue
                # Derived from the configured template, not written again: the
                # host lives in `submissions_url_template` and a second copy of
                # it here is how one of two URLs gets updated.
                base = url.rsplit("/", 1)[0]
                try:
                    older = await client.get(f"{base}/{name}", headers=headers)
                    older.raise_for_status()
                    filtered += self._block_to_filings(
                        older.json(), ticker, cik, start_date)
                except Exception as exc:  # noqa: BLE001 - one batch, not the run
                    # Fails open per batch: an older batch that cannot be read
                    # loses history, it does not lose the ticker.
                    logger.warning(
                        "[SEC] %s: older batch %s could not be read (%s); "
                        "its filings are missing from this run.",
                        ticker, name, exc)

            return filtered

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error processing {ticker} (CIK: {cik}): {e}")
            raise RuntimeError(f"Failed to fetch SEC filings for {ticker} ({cik})") from e
