import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

logger = logging.getLogger(__name__)

def _configure_yfinance_cache() -> None:
    cache_dir = Path("data/cache/yfinance").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except AttributeError:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Installed yfinance version does not support cache location override.")
    except Exception as e:
        logger.warning(f"Could not configure yfinance cache directory '{cache_dir}': {e}", exc_info=True)
        raise RuntimeError(f"Could not configure yfinance cache directory '{cache_dir}'") from e

class YFCollector(BaseCollector):
    """
    Collects and caches historical price data from Yahoo Finance.
    """
    collector_type = "yahoo_finance"
    data_type = "market_data"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, **kwargs)
        self.timeframes = self.configs.get('timeframes', {})
        if not self.timeframes:
            self.logger.warning("'timeframes' not configured. Collector will not be able to gather data.")

    async def run(self, tickers: List[str], end_date: Optional[datetime] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Asynchronously downloads data, filters for new records, and saves them to the database.
        Accepts an optional end_date for deterministic testing.
        """
        if not self.timeframes or not tickers:
            self.logger.info("No tickers or timeframes to collect. Skipping.")
            return []

        end_date = end_date or datetime.now()
        table_name = self.configs.get('table_name', 'market_data_raw')

        # Cache Check
        cache_key = f"{self.__class__.__name__}_run_{self.collector_type}"
        cache_params = {
            "tickers": sorted(tickers),
            "timeframes": self.timeframes,
            "end_date": end_date.isoformat()
        }

        if self.cache_manager and self.db_manager:
            cached_data = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached_data is not None:
                self.logger.info(f"Checking cached data for {len(tickers)} tickers against database.")
                df_cached = pd.DataFrame(cached_data) if isinstance(cached_data, list) else cached_data
                new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                
                if new_from_cache.empty:
                    self.logger.info("Cache hit, but all records are already in the database.")
                    return []
                
                self.logger.info(f"Returning {len(new_from_cache)} new records found in cache.")
                return new_from_cache.to_dict('records')

        # Use reference_now from kwargs if provided for stable testing, otherwise datetime.now()
        reference_now = kwargs.get('reference_now', datetime.now())
        
        self.logger.info(f"Starting collection for {len(tickers)} tickers. End date: {end_date.isoformat()}")

        tasks = []
        for interval, params in self.timeframes.items():
            period = params.get('period')
            start_date = self._calculate_start_date(end_date, period)

            if not start_date:
                continue

            if (interval.endswith('m') or interval.endswith('h')):
                # Yahoo Finance limit for intraday: roughly 60 days from reference point
                limit_date = reference_now - timedelta(days=58)
                if start_date < limit_date:
                    self.logger.warning(f"Interval '{interval}' is intraday and start_date {start_date} is too old. Adjusting to {limit_date}")
                    start_date = limit_date
                
                # Ensure start is before end after adjustment
                if start_date >= end_date:
                    self.logger.warning(f"Adjusted start_date for {interval} is after end_date. Setting end_date to now.")
                    end_date = reference_now

            task = asyncio.to_thread(self._blocking_download, tickers, interval, start_date, end_date)
            tasks.append(task)

        all_price_data = []
        results_from_threads = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results_from_threads):
            if isinstance(result, list) and result:
                all_price_data.extend(result)
        
        if not all_price_data:
            self.logger.info("No data collected from Yahoo Finance.")
            return []

        self.logger.info(f"Collected {len(all_price_data)} total data points from API.")

        df_to_check = pd.DataFrame(all_price_data)
        new_records_df = self.db_manager.filter_new_records(table_name, df_to_check)

        if new_records_df.empty:
            self.logger.info("No new records to save. All collected data is already cached.")
            return []

        self.logger.info(f"Found {len(new_records_df)} new records to save.")
        self.db_manager.upsert(table_name, new_records_df, unique_on=['hash'])
        
        result = new_records_df.to_dict('records')
        if self.cache_manager:
            self.cache_manager.set(cache_key, result, cache_params, ttl=self.configs.get('cache_ttl', 3600), namespace="collectors")
            
        return result

    def _calculate_start_date(self, end_date: datetime, period: str) -> Optional[datetime]:
        try:
            if 'y' in period:
                return end_date - timedelta(days=int(period.replace('y', '')) * 365)
            if 'd' in period:
                return end_date - timedelta(days=int(period.replace('d', '')))
            self.logger.error(f"Unsupported period format: {period}")
        except (ValueError, TypeError):
            self.logger.error(f"Invalid period format: {period}")
        return None

    def _blocking_download(self, tickers: List[str], interval: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Started blocking download for interval '{interval}'.")
        all_ticker_data = []
        for ticker in tickers:
            df = self._single_ticker_download_with_retry(ticker, interval, start_date, end_date)
            if not df.empty:
                processed_data = self._process_single_ticker_dataframe(df, ticker, interval)
                all_ticker_data.extend(processed_data)
        return all_ticker_data

    def _single_ticker_download_with_retry(self, ticker: str, interval: str, start_date: datetime, end_date: datetime, retries: int = 3, delay: int = 5) -> pd.DataFrame:
        _configure_yfinance_cache()
        last_error = None
        for attempt in range(retries):
            try:
                # Use objects directly for intraday or precise strings
                df = yf.download(
                    tickers=ticker,
                    interval=interval,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if not df.empty:
                    self.logger.info(f"Successfully downloaded {len(df)} rows for {ticker}/{interval}")
                    return df
                self.logger.warning(f"No data for {ticker}/{interval} on attempt {attempt + 1}/{retries}. Retrying in {delay}s.")
            except (ValueError, TypeError, Exception) as e:
                self.logger.error(f"Error downloading {ticker}/{interval} on attempt {attempt + 1}/{retries}: {e}", exc_info=True)
                last_error = e
            
            if attempt < retries - 1:
                time.sleep(delay)

        self.logger.error(f"Failed to download data for {ticker}/{interval} after {retries} attempts.")
        raise RuntimeError(f"Data download failed for {ticker}/{interval} after {retries} attempts: {last_error}") from last_error

    def _process_single_ticker_dataframe(self, df: pd.DataFrame, ticker: str, interval: str) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        
        # Flatten MultiIndex PROPERLY
        if isinstance(df.columns, pd.MultiIndex):
            # Get the first level (Price names: Close, High, Low, Open, Volume)
            df.columns = df.columns.get_level_values(0)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Flattened MultiIndex columns for {ticker}/{interval}")

        # Ensure index name is set before reset for clear column naming
        if df.index.name is None:
            df.index.name = 'datetime'
        
        df = df.reset_index()
        
        # Remove duplicated columns if any (e.g. after MultiIndex flattening)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Lowercase column names AFTER flattening
        df.rename(columns={col: str(col).lower() for col in df.columns}, inplace=True)
        
        # Identify date column
        date_candidates = ['datetime', 'date', 'timestamp']
        date_col = next((c for c in date_candidates if c in df.columns), None)
        
        if date_col:
            df.rename(columns={date_col: 'datetime'}, inplace=True)
        else:
            raise RuntimeError(f"Could not find date/datetime column for {ticker}/{interval}. Available: {df.columns.tolist()}")
        
        # Parse datetime with error handling
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            nat_count = df['datetime'].isna().sum()
            if nat_count > 0:
                self.logger.warning(f"Found {nat_count} NaT values in datetime for {ticker}/{interval}")
                # Remove rows with NaT datetime
                df = df[df['datetime'].notna()]
                if df.empty:
                    raise ValueError(f"All datetime values are NaT for {ticker}/{interval}")
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Failed to parse datetime for {ticker}/{interval}: {e}", exc_info=True)
            raise RuntimeError(f"Datetime parsing failed for {ticker}/{interval}: {e}") from e
        
        df['ticker'] = ticker
        df['interval'] = interval
        
        required_cols = ['datetime', 'ticker', 'interval']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for {ticker}/{interval}. Cannot generate hash.")
        
        # Consistent hash generation using isoformat with microsecond precision
        df['hash'] = df.apply(lambda row: hashlib.sha256(f"{row['datetime'].strftime('%Y-%m-%dT%H:%M:%S.%f%z')}{row['ticker']}{row['interval']}".encode()).hexdigest(), axis=1)
        
        self.logger.info(f"✅ Processed {len(df)} rows for {ticker}/{interval} (after NaT removal)")
        
        return df.to_dict('records')
