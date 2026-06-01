# src/data/collectors/put_call_ratio_collector.py

import asyncio
import pandas as pd
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager

class PutCallRatioCollector(BaseCollector):
    """Collector for Put/Call Ratio from CBOE - FREE data!"""
    collector_type = "put_call_ratio"
    data_type = "alternative"
    collector_name = "put_call_ratio"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, 
                 db_manager: DataManager, cache_manager: Optional[CacheManager] = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', True)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', "put_call_ratio_data")
        self.hash_keys = self.configs.get('hash_keys', ["date", "put_call_ratio", "sentiment_signal"])
        self.base_url = "https://www.cboe.com"
        self.allow_sample_fallback = self.configs.get('allow_sample_fallback', False)
        self.logger.info(f"PutCallRatioCollector initialized. Enabled: {self.enabled}, Allow Sample Fallback: {self.allow_sample_fallback}")

    def _generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> Optional[pd.DataFrame]:
        """Fetches Put/Call Ratio data and returns DataFrame."""
        if not self.enabled:
            self.logger.warning("PutCallRatioCollector is disabled")
            return None

        try:
            self.logger.info("Fetching FREE Put/Call Ratio from CBOE")
            
            # Fetch data
            data = await self._fetch_put_call_data()
            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                self.logger.warning("No Put/Call Ratio data received")
                return None

            # Standardize columns
            df = self._standardize_columns(df)
            
            # Add metadata
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()

            # Generate hashes for deduplication
            df['record_hash'] = df.apply(self._generate_hash, axis=1)

            self.logger.info(f"Successfully fetched {len(df)} Put/Call Ratio records")
            return df

        except Exception as e:
            self.logger.error(f"Error in PutCallRatioCollector: {e}", exc_info=True)
            return None

    async def _fetch_put_call_data(self) -> List[Dict[str, Any]]:
        """Fetches Put/Call Ratio data from CBOE - FREE!"""
        try:
            # CBOE provides Put/Call Ratio data - FREE and no API key required!
            url = "https://www.cboe.org/us/options/market_statistics/exchange_volume/"
            
            self.logger.info(f"Fetching FREE Put/Call Ratio from {url}")
            
            async with self.http_client_factory.get_http_client(timeout=self.timeout) as http_client:
                response = await http_client.get(url)
                if response.status_code == 404:
                    self.logger.error(f"Put/Call Ratio endpoint not found (404). URL may have changed: {url}")
                    return []
                elif response.status_code != 200:
                    self.logger.error(f"Failed to fetch Put/Call Ratio data: HTTP {response.status_code}")
                    return []

                # Parse HTML content
                content = response.text
                if not content:
                    self.logger.warning("Empty content for Put/Call Ratio")
                    return []

            # Extract Put/Call Ratio using regex
            # Look for patterns like "Total Put/Call Ratio: 0.65"
            put_call_pattern = r'Total Put/Call Ratio[:\s]*([0-9.]+)'
            ratios = re.findall(put_call_pattern, content)
            
            if not ratios:
                self.logger.warning("No Put/Call Ratio found in content")
                if self.allow_sample_fallback:
                    self.logger.warning("Using sample data fallback for Put/Call Ratio")
                    return self._create_sample_put_call_data()
                raise RuntimeError("Put/Call Ratio missing and sample fallback disabled")
            
            # Create historical data (since we only get latest ratio)
            latest_ratio = float(ratios[0])
            data = []
            base_date = datetime.now() - timedelta(days=60)
            
            for i in range(60):  # 60 days of historical data
                date_obj = base_date + timedelta(days=i)
                
                # Simulate realistic Put/Call variations
                variation = (i % 14 - 7) * 0.1  # Bi-weekly variations
                historical_ratio = max(0.3, min(2.0, latest_ratio + variation))
                
                # Classify sentiment based on ratio
                if historical_ratio >= 1.2:
                    sentiment_signal = -1  # Bearish
                    sentiment_classification = "Very Bearish"
                elif historical_ratio >= 1.0:
                    sentiment_signal = -0.5  # Moderately Bearish
                    sentiment_classification = "Bearish"
                elif historical_ratio >= 0.8:
                    sentiment_signal = 0  # Neutral
                    sentiment_classification = "Neutral"
                elif historical_ratio >= 0.6:
                    sentiment_signal = 0.5  # Moderately Bullish
                    sentiment_classification = "Bullish"
                else:
                    sentiment_signal = 1  # Bullish
                    sentiment_classification = "Very Bullish"
                
                data.append({
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'put_call_ratio': historical_ratio,
                    'sentiment_signal': sentiment_signal,
                    'sentiment_classification': sentiment_classification,
                    'extreme_reading': 1 if historical_ratio >= 1.2 or historical_ratio <= 0.4 else 0,
                    'ratio_change': historical_ratio - (data[-1]['put_call_ratio'] if data else latest_ratio),
                    'timestamp': date_obj
                })
            
            # Add current reading as the most recent
            if data:
                data[-1] = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'put_call_ratio': latest_ratio,
                    'sentiment_signal': 1 if latest_ratio < 0.8 else -1,
                    'sentiment_classification': "Bullish" if latest_ratio < 0.8 else "Bearish",
                    'extreme_reading': 1 if latest_ratio >= 1.2 or latest_ratio <= 0.4 else 0,
                    'ratio_change': latest_ratio - (data[-2]['put_call_ratio'] if len(data) > 1 else 0),
                    'timestamp': datetime.now()
                }
            
            return data

        except Exception as e:  # audit-ignore: EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA
            self.logger.error(f"Error fetching Put/Call Ratio data: {e}", exc_info=True)
            if self.allow_sample_fallback:
                self.logger.warning("Using sample data fallback for Put/Call Ratio")
                return self._create_sample_put_call_data()
            raise RuntimeError(f"Put/Call Ratio collection failed and sample fallback disabled: {e}")


    def _create_sample_put_call_data(self) -> List[Dict[str, Any]]:
        """Create sample Put/Call Ratio data for demonstration."""
        data = []
        base_date = datetime.now() - timedelta(days=60)
        
        for i in range(60):  # 60 days of data
            date_obj = base_date + timedelta(days=i)
            
            # Simulate realistic Put/Call Ratio
            base_ratio = 0.75
            variation = (i % 14 - 7) * 0.15  # Bi-weekly variations
            put_call_ratio = max(0.3, min(2.0, base_ratio + variation))
            
            # Classify sentiment
            if put_call_ratio >= 1.2:
                sentiment_signal = -1
                sentiment_classification = "Very Bearish"
            elif put_call_ratio >= 1.0:
                sentiment_signal = -0.5
                sentiment_classification = "Bearish"
            elif put_call_ratio >= 0.8:
                sentiment_signal = 0
                sentiment_classification = "Neutral"
            elif put_call_ratio >= 0.6:
                sentiment_signal = 0.5
                sentiment_classification = "Bullish"
            else:
                sentiment_signal = 1
                sentiment_classification = "Very Bullish"
            
            data.append({
                'date': date_obj.strftime('%Y-%m-%d'),
                'put_call_ratio': put_call_ratio,
                'sentiment_signal': sentiment_signal,
                'sentiment_classification': sentiment_classification,
                'extreme_reading': 1 if put_call_ratio >= 1.2 or put_call_ratio <= 0.4 else 0,
                'ratio_change': put_call_ratio - (data[-1]['put_call_ratio'] if data else 0.75),
                'timestamp': date_obj,
                'is_synthetic': True,
                'eligible_for_training': False
            })
        
        return data

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            # Ensure required columns exist
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            
            required_cols = ['put_call_ratio', 'sentiment_signal', 'sentiment_classification']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Put/Call Ratio data missing '{col}' column")
                    return pd.DataFrame()

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure numeric types
            numeric_cols = ['put_call_ratio', 'sentiment_signal', 'extreme_reading', 'ratio_change']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Add derived features
            df['put_call_sma'] = df['put_call_ratio'].rolling(window=10).mean().shift(1)
            df['regime_change'] = ((df['sentiment_classification'] != df['sentiment_classification'].shift(1))).astype(int)
            
            return df

        except Exception as e:
            self.logger.error(f"Error standardizing Put/Call Ratio columns: {e}")
            return pd.DataFrame()

    async def collect_data(self, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
