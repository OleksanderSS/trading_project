import asyncio
import hashlib
from datetime import datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class EconomicCalendarCollector(BaseCollector):
    """Fetches economic calendar records mapping Investing.com HTTP payloads data structures string bounds bounds layers limits."""
    collector_type = 'economic_calendar'
    data_type = 'economic'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)

    async def run(self, **kwargs) -> pd.DataFrame | None:
        table_name = self.configs.get('table_name', 'economic_calendar')
        cache_key = f'{self.__class__.__name__}_run'
        
        raw_data = await self.fetch_raw_data()
        if not raw_data:
            return None
            
        df = pd.DataFrame(raw_data)
        if df.empty:
            return None
            
        df['hash'] = df.apply(lambda row: hashlib.sha256(
            f"{row.get('date', '')}|{row.get('title', '')}|{row.get('country', '')}".encode()
        ).hexdigest(), axis=1)
        
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info('[EconCalendar] No new events found.')
            return None
            
        self.db_manager.upsert(table_name, new_df, unique_on=['hash'])
        self.logger.info(f'[EconCalendar] Inserted {len(new_df)} new events.')
        return new_df

    async def run(self, tickers: list[str] | None = None, **kwargs) -> list[dict[str, Any]]:
        # Using ForexFactory free JSON calendar
        api_url = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        try:
            client = await self.http_client_factory.get_http_client()
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            records = []
            
            for item in data:
                # Filter by country if configured
                countries = [c.upper() for c in self.configs.get('countries', ['US', 'EUR', 'GBP'])]
                if item.get('country') not in countries:
                    continue
                    
                records.append({
                    'timestamp': item.get('date'),
                    'country': item.get('country'),
                    'impact': item.get('impact'),
                    'event': item.get('title'),
                    'actual': item.get('actual', ''),
                    'forecast': item.get('forecast', ''),
                    'previous': item.get('previous', '')
                })
                
            self.logger.info(f'[EconCalendar] Fetched {len(records)} events from ForexFactory.')
            return records
            
        except Exception as e:
            self.logger.error(f'Failed to fetch economic calendar: {e}')
            return []

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        data = await self.run(tickers=None, **kwargs)
        return data if data else None
