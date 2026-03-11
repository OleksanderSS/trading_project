import asyncio
import pandas as pd
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager

class FredCollector(BaseCollector):
    """Collector for fetching economic data from the Federal Reserve Economic Data (FRED)."""
    collector_type = "fred"
    data_type = "macro_data"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, cache_manager: Optional[CacheManager] = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.timeout = self.configs.get('timeout', 20.0)
        period_str = self.configs.get('params', {}).get('period', '1y') 
        self.start_date = self._calculate_start_date(period_str)
        self.hash_keys = self.configs.get('hash_keys', ["date", "series_id", "value"])
        self.logger.info(f"FredCollector configured to fetch data from {self.start_date} onwards.")

    def _calculate_start_date(self, period: str) -> str:
        if 'y' in period:
            years = int(period.replace('y', ''))
            return (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        elif 'd' in period:
            days = int(period.replace('d', ''))
            return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        else:
            self.logger.warning(f"Unsupported period format for FRED: {period}. Defaulting to 1 year.")
            return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    def _generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> Optional[pd.DataFrame]:
        """Fetches data from FRED and filters for new records using cache and DB."""
        import os
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            self.logger.error("FRED_API_KEY environment variable not set.")
            return None

        series_ids = self.configs.get('params', {}).get('series_ids', [])
        if not series_ids:
            self.logger.warning("No series_ids specified for FRED. Skipping collection.")
            return None

        client = self.http_client_factory.get_http_client()
        tasks = [self._fetch_series(series_id, client, api_key) for series_id in series_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_series_data = []
        for res in results:
            if isinstance(res, list):
                all_series_data.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(f"Error fetching FRED series: {res}")

        if not all_series_data:
            return None

        df = pd.DataFrame(all_series_data)
        df['hash'] = df.apply(self._generate_hash, axis=1)

        # 1. Filter by CacheManager (if available)
        if self.cache_manager:
            is_new = df['hash'].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("All collected FRED records are already in cache.")
                return None

        # 2. Filter by Database
        table_name = self.configs.get('table_name', 'fred_data')
        new_records_df = self.db_manager.filter_new_records(table_name, df)

        if new_records_df.empty:
            # Update cache for the ones we checked to avoid DB hits next time
            if self.cache_manager:
                for h in df['hash']:
                    self.cache_manager.set(h, True)
            self.logger.info("No new FRED records after DB filtering.")
            return None

        self.logger.info(f"Found {len(new_records_df)} new FRED records.")
        
        # We don't save here, the stage will do it. 
        # But we mark them in cache as "seen" so the stage doesn't have to (or can)
        if self.cache_manager:
            for h in new_records_df['hash']:
                self.cache_manager.set(h, True)

        return new_records_df

    async def _fetch_series(self, series_id: str, client, api_key:str) -> List[Dict[str, Any]]:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={self.start_date}"
        try:
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            observations = data.get('observations', [])
            for obs in observations:
                obs['series_id'] = series_id
            return observations
        except Exception as e:
            self.logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            return []