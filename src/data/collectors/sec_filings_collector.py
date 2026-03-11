# src/data/collectors/sec_filings_collector.py

import asyncio
import httpx
import logging
import json
import pandas as pd
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class SECFilingsCollector(BaseCollector):
    """
    Collects company filings from the SEC EDGAR API.
    CIK codes are dynamically loaded from `assets.yaml`.
    """
    collector_type = "sec_filings"
    data_type = "fundamental"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, cache_manager: Optional[CacheManager] = None, **kwargs):
        super().__init__(configs, http_client_factory, cache_manager=cache_manager, **kwargs)
        self.submissions_url_template = self.configs.get("submissions_url_template")
        self.hash_keys = self.configs.get('hash_keys', ["accessionNumber", "cik"])
        if not self.submissions_url_template:
            raise ValueError("'submissions_url_template' must be specified in the configuration.")
        self._cik_map = None # Cache for CIK map
        self.config_manager = kwargs.get('config_manager')

    def _calculate_start_date(self, period: str, run_date: datetime) -> datetime:
        days = 0
        if 'y' in period:
            days = int(period.replace('y', '')) * 365
        elif 'd' in period:
            days = int(period.replace('d', ''))
        else:
            logger.warning(f"Unsupported period format for SEC Filings: {period}. Defaulting to 60 days.")
            days = 60
        return run_date - timedelta(days=days)

    def _get_cik_map(self) -> Dict[str, str]:
        """Loads and caches the CIK map from `assets.yaml`."""
        if self._cik_map is None:
            try:
                assets_config = self.config_manager.get_config('assets')
                details = assets_config.get('details', {})
                self._cik_map = {
                    ticker: str(data['cik'])
                    for ticker, data in details.items()
                    if 'cik' in data
                }
                logger.info(f"Loaded CIK map for {len(self._cik_map)} tickers from assets.yaml.")
            except Exception as e:
                logger.error(f"Failed to load CIK map from assets.yaml: {e}", exc_info=True)
                self._cik_map = {}
        return self._cik_map

    async def run(self, tickers: List[str], **kwargs) -> Optional[pd.DataFrame]:
        """
        Retrieves filings for the given list of tickers.
        """
        if not tickers:
            logger.warning("No tickers provided for SEC filings collection. Skipping.")
            return None
        
        run_date = kwargs.get('run_date', datetime.now())
        period_str = self.configs.get('params', {}).get('period', '60d')
        start_date = self._calculate_start_date(period_str, run_date)
        logger.info(f"SECFilingsCollector configured to fetch filings from {start_date} onwards.")

        cik_map = self._get_cik_map()
        
        valid_ciks = {ticker: str(cik_map.get(ticker.upper())).zfill(10) for ticker in tickers if ticker.upper() in cik_map}

        if not valid_ciks:
            logger.warning("No valid CIKs found for the provided tickers.")
            return None

        logger.info(f"Starting SEC filings collection for {len(valid_ciks)} tickers.")
        all_filings: List[Dict[str, Any]] = []

        async with self.http_client_factory.get_http_client() as client:
            tasks = [self._fetch_filings_for_cik(ticker, cik, client, start_date) for ticker, cik in valid_ciks.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
            for i, res in enumerate(results):
                if isinstance(res, list):
                    all_filings.extend(res)
                elif isinstance(res, Exception):
                    ticker = list(valid_ciks.keys())[i]
                    logger.error(f"Error fetching filings for {ticker}: {res}")

        if not all_filings:
            logger.info("No raw SEC filings received.")
            return None

        logger.info(f"Total {len(all_filings)} raw SEC filings received.")
        
        filings_df = pd.DataFrame(all_filings)
        
        for key in self.hash_keys:
            if key in filings_df.columns:
                filings_df[key] = filings_df[key].astype(str)

        filings_df['hash'] = filings_df.apply(lambda row: hashlib.sha256("".join(str(row[key]) for key in self.hash_keys).encode()).hexdigest(), axis=1)

        if self.cache_manager:
            is_new = filings_df['hash'].apply(lambda h: self.cache_manager.get(h) is None)
            new_records_df = filings_df[is_new].copy()
            if new_records_df.empty:
                logger.info("All collected SEC Filings records are already in cache.")
                return None
        else:
            new_records_df = filings_df

        if self.cache_manager:
            for h in new_records_df['hash']:
                self.cache_manager.set(h, True)

        self.logger.info(f"Found {len(new_records_df)} new SEC filings to save.")

        return new_records_df

    async def _fetch_filings_for_cik(self, ticker: str, cik: str, client: httpx.AsyncClient, start_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetches, processes, and normalizes filings for a single CIK.
        """
        url = self.submissions_url_template.format(cik=cik)
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            recent_filings = data.get('filings', {}).get('recent', {})
            if not recent_filings or 'accessionNumber' not in recent_filings:
                return []

            keys = recent_filings.keys()
            filings_list = [{key: recent_filings[key][i] for key in keys} for i in range(len(recent_filings['accessionNumber']))]
            
            filtered_by_date = []
            for filing in filings_list:
                try:
                    filing_date = datetime.strptime(filing['filingDate'], '%Y-%m-%d')
                    if filing_date >= start_date:
                        filing['ticker'] = ticker
                        filing['cik'] = cik
                        for key, value in filing.items():
                            if isinstance(value, list):
                                filing[key] = json.dumps(value)
                        filtered_by_date.append(filing)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse date for filing {filing.get('accessionNumber')}: {e}")
                    continue
            
            return filtered_by_date
        except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Error processing {ticker} (CIK: {cik}): {e}")
            return []