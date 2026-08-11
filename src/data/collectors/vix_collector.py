# src/data/collectors/vix_collector.py

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

def _configure_yfinance_cache() -> None:
    cache_dir = Path("data/cache/yfinance").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yfinance as yf
        yf.set_tz_cache_location(str(cache_dir))
    except AttributeError as e:
        logger.debug(f"yfinance does not support set_tz_cache_location: {e}")
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Failed to set yfinance cache location: {e}", exc_info=True)

class VIXCollector(BaseCollector):
    """Collector for VIX Volatility Index - FREE data from Yahoo Finance!"""
    collector_type = "vix"
    data_type = "alternative"
    collector_name = "vix"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory,
                 db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', True)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', "vix_data")
        # Default must name real columns: _standardize_columns produces
        # vix_close and rejects a frame without it. "vix_current" existed in
        # neither the collector nor the table, so every row's hash was built
        # from an empty string in that slot, and the same name reaching
        # DataManager as `unique_on` broke this table's unique index and its
        # duplicate check outright.
        self.hash_keys = self.configs.get('hash_keys', ["date", "vix_close", "volatility_regime"])

        # Merge parameters from configuration structure
        params = self.configs.get('params', {})
        self.period = params.get('period', '30d')
        self.interval = params.get('interval', '1d')
        self.ticker = self.configs.get('ticker', '^VIX')

        self.logger.info(f"VIXCollector initialized. Enabled: {self.enabled}, Period: {self.period}, Interval: {self.interval}")

    def generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> pd.DataFrame | None:
        """Fetches VIX data and returns DataFrame."""
        if not self.enabled:
            self.logger.warning("VIXCollector is disabled")
            return None

        try:
            self.logger.info("Fetching FREE VIX data from Yahoo Finance")

            # Fetch data
            data = await self._fetch_vix_data()
            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                self.logger.warning("No VIX data received")
                return None

            # Standardize columns
            df = self._standardize_columns(df)

            # Add metadata
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()

            # Generate hashes for deduplication
            df['record_hash'] = df.apply(self.generate_hash, axis=1)

            self.logger.info(f"Successfully fetched {len(df)} VIX records")
            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error in VIXCollector: {e}")
            raise RuntimeError("VIX collection failed") from e

    async def _fetch_vix_data(self) -> list[dict[str, Any]]:
        """Fetches VIX data from Yahoo Finance - FREE!"""
        try:
            # Use Yahoo Finance for VIX data - FREE and no API key required!
            import yfinance as yf
            _configure_yfinance_cache()

            self.logger.info("Fetching VIX data from Yahoo Finance")

            # Download VIX data for last 60 days
            vix_ticker = yf.Ticker("^VIX")
            hist = vix_ticker.history(period="60d", interval="1d")

            if hist.empty:
                self.logger.warning("No VIX data from Yahoo Finance")
                return []

            # Process historical data
            data = []
            for idx, (date, row) in enumerate(hist.iterrows()):
                vix_close = row['Close']
                vix_high = row['High']
                vix_low = row['Low']
                vix_volume = row['Volume']

                # Calculate volatility regime using only historical data up to current row
                # Use .iloc[:idx+1] to get only data up to current row to avoid lookahead
                hist_up_to_now = hist.iloc[:idx+1]
                if len(hist_up_to_now) >= 20:
                    vix_sma = hist_up_to_now['Close'].rolling(window=20).mean().iloc[-1]
                else:
                    vix_sma = vix_close
                volatility_regime = 'high' if vix_close > vix_sma else 'low'

                # Calculate VIX percentiles using only historical data up to current row
                # This prevents lookahead bias where future data influences historical percentiles
                if len(hist_up_to_now) >= 20:
                    vix_percentile_20 = hist_up_to_now['Close'].quantile(0.2)
                    vix_percentile_80 = hist_up_to_now['Close'].quantile(0.8)
                else:
                    # Not enough data for meaningful percentiles, use current value
                    vix_percentile_20 = vix_close
                    vix_percentile_80 = vix_close

                # Classify VIX level
                if vix_close >= 30:
                    vix_classification = "Extreme Fear"
                elif vix_close >= 25:
                    vix_classification = "Fear"
                elif vix_close >= 20:
                    vix_classification = "Neutral"
                elif vix_close >= 15:
                    vix_classification = "Greed"
                else:
                    vix_classification = "Extreme Greed"

                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'vix_open': row['Open'],
                    'vix_high': vix_high,
                    'vix_low': vix_low,
                    'vix_close': vix_close,
                    'vix_volume': vix_volume,
                    'vix_sma_20': vix_sma,
                    'volatility_regime': volatility_regime,
                    'vix_classification': vix_classification,
                    'vix_percentile_20': vix_percentile_20,
                    'vix_percentile_80': vix_percentile_80,
                    'vix_range': vix_high - vix_low,
                    'vix_change': vix_close - hist_up_to_now['Close'].shift(1).iloc[-1] if len(hist_up_to_now) > 1 else 0,
                    'extreme_volatility': 1 if vix_close >= 30 or vix_close <= 12 else 0,
                    'timestamp': date
                })

            return data

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error fetching VIX data: {e}")
            raise RuntimeError("Failed to fetch VIX data") from e

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            # Ensure required columns exist
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

            required_cols = ['vix_close', 'volatility_regime', 'vix_classification']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"VIX data missing '{col}' column")
                    return pd.DataFrame()

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])

            # Ensure numeric types
            numeric_cols = ['vix_open', 'vix_high', 'vix_low', 'vix_close', 'vix_volume',
                           'vix_sma_20', 'vix_percentile_20', 'vix_percentile_80',
                           'vix_range', 'vix_change', 'extreme_volatility']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            # Add derived features
            df['vix_signal'] = df['volatility_regime'].apply(lambda x: 1 if x == 'high' else -1)
            df['vix_zscore'] = (df['vix_close'] - df['vix_close'].rolling(20).mean().shift(1)) / df['vix_close'].rolling(20).std().shift(1)

            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error standardizing VIX columns: {e}")
            return pd.DataFrame()

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
