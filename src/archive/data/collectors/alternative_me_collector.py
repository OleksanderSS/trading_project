"""
Alternative.me Fear & Greed Index Collector
Alternative source for Fear & Greed data when CNN API is unavailable
"""

import hashlib
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from src.data.collectors.base_collector import BaseCollector


class AlternativeMeCollector(BaseCollector):
    """Collector for Fear & Greed Index data from Alternative.me"""
    collector_type = "alternative_me"
    data_type = "alternative"
    collector_name = "alternative_me"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory,
                 db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', True)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', "fear_greed_data")
        self.hash_keys = self.configs.get('hash_keys', ["date", "value"])
        self.base_url = "https://api.alternative.me"
        self.logger.info(f"AlternativeMeCollector initialized. Enabled: {self.enabled}")

    def _generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> pd.DataFrame | None:
        """Fetches Fear & Greed data from Alternative.me and returns DataFrame."""
        if not self.enabled:
            self.logger.warning("AlternativeMeCollector is disabled")
            return None

        try:
            self.logger.info("Fetching Fear & Greed data from Alternative.me API")

            # Fetch data
            data = await self._fetch_fear_greed_data()
            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                self.logger.warning("No Fear & Greed data received from Alternative.me")
                return None

            # Generate hash
            df['hash'] = df.apply(self._generate_hash, axis=1)

            # Standardize columns
            df = self._standardize_columns(df)

            self.logger.info(f"Successfully fetched {len(df)} Fear & Greed records from Alternative.me")
            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in AlternativeMeCollector: {e}", exc_info=True)
            raise RuntimeError("Alternative.me collection failed") from e

    async def _fetch_fear_greed_data(self) -> list[dict[str, Any]]:
        """Fetches data from Alternative.me API."""
        try:
            client = await self.http_client_factory.get_http_client(
                timeout=self.timeout
            )

            # Alternative.me Fear & Greed API endpoint (V1 works) - increased limit to 100
            url = f"{self.base_url}/fng/?limit=100"

            response = await client.get(url)
            if response.status_code == 404:
                self.logger.error(f"Alternative.me Fear & Greed API endpoint not found (404). URL may have changed: {url}")
                return []
            elif response.status_code != 200:
                self.logger.error(f"Failed to fetch Alternative.me data: HTTP {response.status_code}")
                return []

            # Alternative.me V1 returns JSON with data as list
            json_data = response.json()

            # Alternative.me V1 structure: {"name": "Fear & Greed Index", "data": [{"value": "13", "value_classification": "Extreme Fear", "timestamp": "1774569600", "time_until_update": "69573"}], "metadata": {"error": None}}
            data = json_data.get('data', [])

            if not data:
                self.logger.warning("Empty data received from Alternative.me API")
                return []

            # Process the data
            processed_data = []

            # Alternative.me V1 returns data as list
            if isinstance(data, list):
                for item in data:
                    processed_data.append(self._process_data_point(item))
            else:
                self.logger.warning(f"Unexpected data format from Alternative.me V1: {type(data)}")
                return []

            return processed_data

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error fetching Alternative.me data: {e}", exc_info=True)
            raise RuntimeError("Failed to fetch Alternative.me data") from e

    def _process_data_point(self, data_point: dict[str, Any]) -> dict[str, Any]:
        """Process a single data point from Alternative.me"""
        try:
            # Alternative.me timestamp is in seconds
            raw_ts = data_point.get('timestamp', 0)
            try:
                timestamp = int(raw_ts)
            except (TypeError, ValueError):
                timestamp = 0

            if timestamp > 0:
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            else:
                date = datetime.now().strftime('%Y-%m-%d')

            return {
                'date': date,
                'value': float(data_point.get('value', 0)),
                'timestamp': datetime.fromtimestamp(timestamp).isoformat() if timestamp > 0 else datetime.now().isoformat(),
                'classification': data_point.get('value_classification', 'Unknown'),
                'source': 'alternative_me'
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error processing Alternative.me data point: {e}")
            raise RuntimeError("Failed to process Alternative.me data point") from e

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            # Ensure required columns exist
            if 'date' not in df.columns:
                if 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

            if 'value' not in df.columns or df.empty:
                self.logger.error("Alternative.me data missing 'value' column or empty")
                return pd.DataFrame()

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])

            # Ensure numeric types
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error standardizing Alternative.me columns: {e}")
            return pd.DataFrame()

    async def collect_data(self, **kwargs) -> pd.DataFrame | None:
        """
        BaseCollector contract: Stage calls BaseCollector.run() which delegates here.
        Keep existing implementation in run() for backward compatibility.
        """
        return await self.run(**kwargs)
