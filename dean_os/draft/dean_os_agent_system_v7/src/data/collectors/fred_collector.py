import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class FredCollector(BaseCollector):
    """Collector for fetching economic data from the Federal Reserve Economic Data (FRED)."""
    collector_type = "fred"
    data_type = "macro_data"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
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

    def _validate_config(self, **kwargs) -> tuple[str | None, list[str] | None]:
        """Validate FRED configuration and return API key and series IDs."""
        import os
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            self.logger.error("FRED_API_KEY environment variable not set.")
            return None, None

        series_ids = self.configs.get('params', {}).get('series_ids', [])
        if not series_ids:
            self.logger.warning("No series_ids specified for FRED. Skipping collection.")
            return api_key, None

        return api_key, series_ids

    def _filter_by_cache(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by cache manager if available."""
        if self.cache_manager:
            is_new = df['hash'].apply(lambda h: self.cache_manager.get(h) is None)
            df = df[is_new].copy()
            if df.empty:
                self.logger.info("All collected FRED records are already in cache.")
                return df
        return df

    def _update_cache(self, df: pd.DataFrame) -> None:
        """Update cache with hashes from DataFrame."""
        if self.cache_manager:
            for h in df['hash']:
                self.cache_manager.set(h, True)

    async def run(self, **kwargs) -> pd.DataFrame | None:
        """Fetches data from FRED and filters for new records using cache and DB."""
        api_key, series_ids = self._validate_config(**kwargs)
        if not api_key or not series_ids:
            return None

        client = await self.http_client_factory.get_http_client()
        async with client:
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
        df = self._filter_by_cache(df)
        if df.empty:
            return None

        # 2. Filter by Database
        table_name = self.configs.get('table_name', 'fred_data')
        new_records_df = self.db_manager.filter_new_records(table_name, df)

        if new_records_df.empty:
            # Update cache for the ones we checked to avoid DB hits next time
            self._update_cache(df)
            self.logger.info("No new FRED records after DB filtering.")
            return None

        self.logger.info(f"Found {len(new_records_df)} new FRED records.")

        # ✅ FIX: save to DB so macro data accumulates and is available for Stage 3
        table_name = self.configs.get('table_name', 'fred_data')
        try:
            self.db_manager.upsert(table_name, new_records_df, unique_on=['hash'])
            self.logger.info(f"Saved {len(new_records_df)} FRED records to '{table_name}'")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Could not save FRED records to DB: {e}")

        self._update_cache(new_records_df)

        return new_records_df

    async def _fetch_series(self, series_id: str, client, api_key:str) -> list[dict[str, Any]]:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={self.start_date}"
        try:
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            observations = data.get('observations', [])
            for obs in observations:
                obs['series_id'] = series_id
            return observations
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            raise RuntimeError(f"Failed to fetch FRED series {series_id}") from e
