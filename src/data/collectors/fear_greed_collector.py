import asyncio
import pandas as pd
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager


class FearGreedCollector(BaseCollector):
    """Collector for CNN Fear & Greed Index - FREE data!"""
    collector_type = 'fear_greed'
    data_type = 'alternative'
    collector_name = 'fear_greed'

    def __init__(self, configs: Dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: Optional
        [CacheManager]=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', True)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', 'fear_greed_data')
        self.hash_keys = self.configs.get('hash_keys', ['date',
            'fear_greed_index', 'classification'])
        self.base_url = 'https://production.datapoint.cloud/api/fear-greed'
        self.logger.info(
            f'FearGreedCollector initialized. Enabled: {self.enabled}')

    def _generate_hash(self, row: pd.Series) ->str:
        """Generates a stable hash for a record."""
        hash_string = '|'.join(str(row.get(key, '')) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) ->Optional[pd.DataFrame]:
        """Fetches Fear & Greed data and returns DataFrame."""
        if not self.enabled:
            self.logger.warning('FearGreedCollector is disabled')
            return None
        try:
            self.logger.info('Fetching Fear & Greed data from CNN Business API'
                )
            data = await self._fetch_fear_greed_data()
            if not data:
                return None
            df = pd.DataFrame(data)
            if df.empty:
                self.logger.warning('No Fear & Greed data received')
                return None
            df = self._standardize_columns(df)
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()
            df['record_hash'] = df.apply(self._generate_hash, axis=1)
            self.logger.info(
                f'Successfully fetched {len(df)} Fear & Greed records')
            return df
        except Exception as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Error in FearGreedCollector: {e}')
            return None

    async def _fetch_fear_greed_data(self) ->List[Dict[str, Any]]:
        """Fetches data from CNN Fear & Greed API."""
        try:
            client = await self.http_client_factory.get_http_client(timeout=self.
                timeout)
            async with client:
                url = f'{self.base_url}/1.0/JSON'
                response = await client.get(url)
                if response.status_code == 404:
                    self.logger.error(
                        f'Fear & Greed API endpoint not found (404). URL may have changed: {url}'
                        )
                    self.logger.error(
                        "This API may have been deprecated or moved. Check CNN's API documentation."
                        )
                    return []
                elif response.status_code != 200:
                    self.logger.error(
                        f'Failed to fetch Fear & Greed data: HTTP {response.status_code}'
                        )
                    return []
                json_data = response.json()
                data = json_data.get('data', [])
                if not data:
                    self.logger.warning(
                        'Empty data received from Fear & Greed API')
                    return []
            processed_data = []
            for item in data:
                try:
                    timestamp = item.get('x')
                    value = item.get('y')
                    if timestamp is not None and value is not None:
                        processed_data.append({'date': pd.to_datetime(
                            timestamp, unit='ms').strftime('%Y-%m-%d'),
                            'value': float(value), 'timestamp': pd.
                            to_datetime(timestamp, unit='ms')})
                except Exception as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.logger.warning(
                        f'Error processing Fear & Greed item: {e}')
                    continue
            return processed_data
        except Exception as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Error fetching Fear & Greed data: {e}',
                exc_info=True)
            return []

    def _standardize_columns(self, df: pd.DataFrame) ->pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            if 'date' not in df.columns:
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp']).dt.strftime(
                        '%Y-%m-%d')
                else:
                    self.logger.error(
                        "Fear & Greed data missing both 'date' and 'timestamp' columns"
                        )
                    return pd.DataFrame()
            if 'value' not in df.columns or df.empty:
                self.logger.error(
                    "Fear & Greed data missing 'value' column or empty")
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
            df['fear_greed_category'] = df['value'].apply(self.
                _categorize_fear_greed)
            df['fear_greed_signal'] = df['value'].apply(self._get_signal)
            return df
        except Exception as e:
            self.logger.error(f'Error standardizing Fear & Greed columns: {e}')
            return pd.DataFrame()

    def _categorize_fear_greed(self, value: float) ->str:
        """Categorize Fear & Greed value."""
        if value < 25:
            return 'extreme_fear'
        elif value < 45:
            return 'fear'
        elif value < 55:
            return 'neutral'
        elif value < 75:
            return 'greed'
        else:
            return 'extreme_greed'

    def _get_signal(self, value: float) ->int:
        """Get trading signal based on Fear & Greed value."""
        if value < 25:
            return 1
        elif value > 75:
            return -1
        else:
            return 0

    async def collect_data(self, **kwargs) ->Optional[List[Dict[str, Any]]]:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
