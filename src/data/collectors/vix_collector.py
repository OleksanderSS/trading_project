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

# Trailing rows behind every derived statistic here: the moving average, the
# percentiles and the z-score all read the same window, so a given trading day
# yields the same numbers no matter when it was collected.
_STAT_WINDOW = 20

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
        #
        # The default is now the DATE alone. VIX is one market-wide series with
        # one reading per trading day, so the date is the whole identity.
        # `vix_close` and `volatility_regime` are content, and putting content
        # in an identity key turns it into a change detector: the regime is
        # derived from a moving average, it flipped between collections of the
        # same day, and each flip stored a second row. 22 of 77 dates carried
        # duplicates. Same defect, different table, as the 273 duplicate bars
        # in market_data_raw.
        self.hash_keys = self.configs.get('hash_keys', ["date"])

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

            # These came from the config until now only in appearance:
            # `self.period` and `self.interval` were read in __init__, printed
            # in the "VIXCollector initialized" line, and then never used --
            # the call below said "60d" and "1d" outright. So the log
            # announced 30d while 60 days were fetched, and changing the
            # config changed nothing but the log.
            #
            # The window matters because `_STAT_WINDOW` is 20: the first 20
            # rows of whatever is fetched have no statistics at all. Over 60
            # days (~41 trading days) that is roughly half the rows collected;
            # over two years (~502) it is about 4%.
            #
            # Widening is safe for the VALUES because the statistics below use
            # a fixed trailing window (`iloc[-_STAT_WINDOW:]`), not "everything
            # that happened to be fetched" -- which is the defect the long
            # comment further down records as already fixed. More history
            # therefore adds rows that have statistics; it does not change the
            # statistics of any row that already had them.
            vix_ticker = yf.Ticker(self.ticker)
            hist = vix_ticker.history(period=self.period, interval=self.interval)

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
                if len(hist_up_to_now) >= _STAT_WINDOW:
                    recent = hist_up_to_now['Close'].iloc[-_STAT_WINDOW:]
                    vix_sma = recent.mean()
                    # Over the SAME trailing window as the mean, not over all
                    # the history that happened to be fetched.
                    #
                    # `hist` is `history(period="60d")`, so its first row is 60
                    # days before the COLLECTION date. Taking a quantile over
                    # everything up to `idx` therefore measured a window whose
                    # start moved with the collection date, and the same
                    # trading day came out differently every time it was
                    # collected. Observed on 2026-06-05: identical vix_close of
                    # 21.51 from the 2026-07-20 and 2026-08-04 runs, with
                    # different vix_percentile_20, _80, vix_sma_20, vix_zscore
                    # and volatility_regime. A feature whose value depends on
                    # when you happened to fetch it cannot be trained on and
                    # served consistently, and this one also flipped
                    # volatility_regime -- which was part of hash_keys, so each
                    # recollection stored a SECOND row for that day. 22 of 77
                    # dates were duplicated this way.
                    vix_percentile_20 = recent.quantile(0.2)
                    vix_percentile_80 = recent.quantile(0.8)
                else:
                    # Substituting the current value for a percentile invents a
                    # statistic: it made every early row read "today sits
                    # exactly at the 20th AND the 80th percentile", and made
                    # volatility_regime 'low' by comparing a value to itself.
                    # Absent is the honest answer, and downstream already
                    # distinguishes missing from neutral.
                    vix_sma = float('nan')
                    vix_percentile_20 = float('nan')
                    vix_percentile_80 = float('nan')
                volatility_regime = (
                    'unknown' if pd.isna(vix_sma)
                    else ('high' if vix_close > vix_sma else 'low')
                )

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
