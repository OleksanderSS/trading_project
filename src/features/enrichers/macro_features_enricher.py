import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger('MacroFeaturesEnricher')
DATETIME64_NS = 'datetime64[ns]'


class MacroFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with macroeconomic indicators from FRED.
    Implements caching to avoid repeated downloads.
    """

    @property
    def name(self) ->str:
        return 'macro_features'

    @property
    def priority(self) ->int:
        """Execution order - run after technical analysis (20) but before NLP (30)"""
        return 27

    def __init__(self, config: dict=None):
        """Initialize with optional config dict from FeatureOrchestrator"""
        super().__init__()
        config_manager = get_current_config()
        self.config = config_manager.get(
            'enrichment.macro_features.macro_fred_series', {})
        if not self.config:
            self.config = config_manager.get('macro_features.macro_fred_series'
                , {})
        self.cache_path = Path('./cache') / 'macro_data.parquet'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config:
            logger.warning(
                "Configuration for macro features ('macro_fred_series') not found."
                )
        else:
            logger.info(
                f'✅ MacroFeaturesEnricher initialized with {len(self.config)} series'
                )

    def _validate_dataframe(self, df: pd.DataFrame) ->pd.DataFrame:
        """Validate and prepare DataFrame for enrichment."""
        if df.empty:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
                logger.info("Converted 'datetime' column to DatetimeIndex")
            else:
                logger.error(
                    "Cannot enrich macro features: no DatetimeIndex or 'datetime' column"
                    )
                return df
        return df

    def _filter_test_ticker(self, df: pd.DataFrame) ->pd.DataFrame:
        """Filter DataFrame for test ticker if in test mode."""
        import json
        config_manager = get_current_config()
        params_path = config_manager.get_runtime_params_path()
        if params_path.exists():
            try:
                with open(params_path) as f:
                    runtime_params = json.load(f)
                test_ticker = runtime_params.get('test_mode', {}).get(
                    'test_ticker')
                if test_ticker and 'ticker' in df.columns:
                    logger.info(
                        f'MacroFeaturesEnricher: filtering for ticker {test_ticker}'
                        )
                    df = df[df['ticker'] == test_ticker]
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f'Could not load runtime params for test ticker filtering: {e}'
                        )
                return df
        return df

    def _prepare_macro_data(self, df: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """Prepare macro data from Stage 1 or load from FRED API."""
        macro_data = kwargs.get('macro_data')
        if macro_data is not None and isinstance(macro_data, pd.DataFrame) and not macro_data.empty:
            logger.info(f'Using macro_data from Stage 1 ({len(macro_data)} rows in long format)')
            # ✅ FIX: Stage 1 gives only NEW records (delta), so pivot and return as-is for merging
            # But we also need to rebuild cache from full DB history
            pivoted = self._pivot_macro_data(macro_data)
            if not pivoted.empty:
                # Try to load full history from cache and merge with new data
                full_cache = self._load_full_macro_from_cache()
                if not full_cache.empty:
                    merged = pd.concat([full_cache, pivoted]).sort_index()
                    merged = merged[~merged.index.duplicated(keep='last')]
                    # Save updated cache
                    try:
                        merged.to_parquet(self.cache_path)
                        logger.info(f'Updated macro cache with {len(pivoted)} new rows → {len(merged)} total')
                    except Exception as e:
                        logger.exception(f'Failed to save macro cache: {e}')
                    return merged
                return pivoted
        # Fallback to FRED API or cache
        logger.info('No macro_data in kwargs, loading from FRED API or cache...')
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index.min()
            end_date = df.index.max()
        elif 'datetime' in df.columns:
            start_date = pd.to_datetime(df['datetime']).min()
            end_date = pd.to_datetime(df['datetime']).max()
        else:
            logger.warning('No datetime index or column found, using default date range')
            start_date = pd.Timestamp('2020-01-01')
            end_date = pd.Timestamp.now()
        # ✅ Strip timezone
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.tz_localize(None)
        return self._load_macro_data(start_date, end_date)

    def _load_full_macro_from_cache(self) -> pd.DataFrame:
        """Load all macro data from cache regardless of date range."""
        if self.cache_path.exists():
            try:
                cached = pd.read_parquet(self.cache_path)
                if not cached.empty:
                    return cached
            except Exception as e:
                logger.warning(f'Failed to load macro cache: {e}')
        return pd.DataFrame()

    def _pivot_macro_data(self, macro_data: pd.DataFrame) ->pd.DataFrame:
        """Pivot macro data from long to wide format."""
        if ('series_id' not in macro_data.columns or 'value' not in
            macro_data.columns):
            return macro_data
        date_col = None
        for col in ['date', 'datetime', 'realtime_start']:
            if col in macro_data.columns:
                date_col = col
                break
        if not date_col:
            logger.warning('No date column found in macro_data for pivoting')
            return macro_data
        macro_data[date_col] = pd.to_datetime(macro_data[date_col])
        macro_pivoted = macro_data.pivot_table(index=date_col, columns=
            'series_id', values='value', aggfunc='last')
        macro_pivoted.columns = [f'FRED_{col}' for col in macro_pivoted.columns
            ]
        logger.info(
            f'Pivoted macro data into {len(macro_pivoted.columns)} FRED columns'
            )
        return macro_pivoted

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """
        Adds macro features to the DataFrame.
        First tries to use macro_data from kwargs (collected in Stage 1),
        then falls back to FRED API if needed.

        Args:
            df: DataFrame with a DatetimeIndex.
            **kwargs: May contain 'macro_data' from Stage 1

        Returns:
            DataFrame with added macro features.
        """
        if df.empty:
            return df
        self.df = df
        df = self._validate_dataframe(df)
        if df.empty:
            return df
        df = self._filter_test_ticker(df)
        start_date = df.index.min()
        end_date = df.index.max()
        unique_dates = len(df.index.unique())
        logger.info(
            f'MacroFeaturesEnricher processing {len(df)} records ({unique_dates} unique dates) from {start_date} to {end_date}'
            )
        macro_data = self._prepare_macro_data(df, **kwargs)
        if macro_data.empty:
            logger.warning('Could not load macro data. Skipping enrichment.')
            return df
        return self._merge_macro_data(df, macro_data)
    def _prepare_macro_index(self, macro_data: pd.DataFrame) ->pd.DataFrame:
        """Prepare macro data index for merging."""
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            if 'datetime' in macro_data.columns:
                macro_data = macro_data.set_index('datetime')
            elif 'date' in macro_data.columns:
                macro_data = macro_data.set_index('date')
        macro_data = macro_data.sort_index()
        macro_data = macro_data[~macro_data.index.duplicated(keep='last')]
        if isinstance(macro_data.index, pd.DatetimeIndex):
            if macro_data.index.tz is not None:
                macro_data.index = macro_data.index.tz_localize(None)
            if macro_data.index.dtype != DATETIME64_NS:
                macro_data.index = macro_data.index.astype(DATETIME64_NS)
        return macro_data

    def _normalize_datetime_column(self, df: pd.DataFrame, datetime_col:
        str='datetime') ->pd.DataFrame:
        """Normalize datetime column timezone and precision."""
        if datetime_col not in df.columns:
            return df
        if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            if hasattr(df[datetime_col].dtype, 'tz') and df[datetime_col
                ].dt.tz is not None:
                df[datetime_col] = df[datetime_col].dt.tz_localize(None)
            if df[datetime_col].dtype != DATETIME64_NS:
                df[datetime_col] = df[datetime_col].astype(DATETIME64_NS)
        return df

    def _merge_with_duplicates(self, df: pd.DataFrame, macro_data: pd.DataFrame
        ) ->pd.DataFrame:
        """Merge data when DataFrame has duplicate index labels (multiple tickers)."""
        logger.info(
            'Detected duplicate index labels (multiple tickers). Using merge instead of reindex.'
            )
        df_reset = df.reset_index()
        macro_reset = macro_data.reset_index()
        datetime_col = None
        for col in ['datetime', 'index', 'date']:
            if col in df_reset.columns:
                datetime_col = col
                break
        if datetime_col is None:
            logger.error('Cannot find datetime column after reset_index')
            return df
        if datetime_col != 'datetime':
            df_reset = df_reset.rename(columns={datetime_col: 'datetime'})
        df_reset = self._normalize_datetime_column(df_reset, 'datetime')
        macro_datetime_col = None
        for col in ['datetime', 'index', 'date']:
            if col in macro_reset.columns:
                macro_datetime_col = col
                break
        if macro_datetime_col and macro_datetime_col != 'datetime':
            macro_reset = macro_reset.rename(columns={macro_datetime_col:
                'datetime'})
        macro_reset = self._normalize_datetime_column(macro_reset, 'datetime')
        df_merged = pd.merge_asof(df_reset.sort_values('datetime'),
            macro_reset.sort_values('datetime'), on='datetime', direction=
            'backward')
        if 'datetime' in df_merged.columns:
            return df_merged.set_index('datetime')
        else:
            logger.error("'datetime' column missing after merge_asof")
            return df_merged

    def _merge_without_duplicates(self, df: pd.DataFrame, macro_data: pd.
        DataFrame) ->pd.DataFrame:
        """Merge data when DataFrame has unique index labels."""
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df_index = df.index.tz_localize(None)
        else:
            df_index = df.index
        if hasattr(macro_data.index, 'tz') and macro_data.index.tz is not None:
            macro_data_index = macro_data.index.tz_localize(None)
        else:
            macro_data_index = macro_data.index
        macro_dates = macro_data_index.unique()
        aligned_macro_data = macro_data.loc[macro_dates].reindex(df_index,
            method='ffill')
        for col in aligned_macro_data.columns:
            if col not in df.columns:
                df[col] = aligned_macro_data[col].values
        return df

    def _post_process_fred_columns(self, df: pd.DataFrame) ->pd.DataFrame:
        """Post-process FRED columns with forward fill and NaN handling."""
        fred_cols = [col for col in df.columns if col.startswith('FRED_')]
        if not fred_cols:
            return df
        for col in fred_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[fred_cols] = df[fred_cols].ffill(limit=60)
        remaining_nans = df[fred_cols].isna().sum()
        if remaining_nans.any():
            logger.warning(
                f'Some FRED columns still have NaN after ffill: {remaining_nans[remaining_nans > 0].to_dict()}'
                )
            df[fred_cols] = df[fred_cols].fillna(df[fred_cols].median())
        return df

    def _merge_macro_data(self, df: pd.DataFrame, macro_data: pd.DataFrame
        ) ->pd.DataFrame:
        """Merge macro data with main DataFrame."""
        logger.info('Joining macro data with the main DataFrame...')
        macro_data = self._prepare_macro_index(macro_data)
        if df.index.duplicated().any():
            df = self._merge_with_duplicates(df, macro_data)
        else:
            df = self._merge_without_duplicates(df, macro_data)
        df = df.ffill()
        df = self._post_process_fred_columns(df)
        logger.info(
            f'Macro features successfully added. Final shape: {df.shape}')
        return df

    def _create_fred_session(self):
        """Create FRED API session with retry logic and timeout."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(total=1, backoff_factor=0.5,
                      status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        # Override request() to enforce 45s timeout per series
        _orig_request = session.request
        def _request_with_timeout(method, url, **kwargs):
            kwargs.setdefault('timeout', 45)
            return _orig_request(method, url, **kwargs)
        session.request = _request_with_timeout

        return session

    def _load_fred_series(self, series_ids, series_names, start_date_str, end_date_str, session):
        """Load FRED series using fast JSON API endpoint (much faster than CSV/DataReader)."""
        api_key = os.getenv('FRED_API_KEY')
        collected = {}
        for sid, sname in zip(series_ids, series_names, strict=True):
            url = (f'https://api.stlouisfed.org/fred/series/observations'
                   f'?series_id={sid}&api_key={api_key}&file_type=json'
                   f'&observation_start={start_date_str}&observation_end={end_date_str}')
            try:
                resp = session.get(url)
                resp.raise_for_status()
                obs = resp.json().get('observations', [])
                valid = [(o['date'], float(o['value'])) for o in obs if o['value'] != '.']
                if valid:
                    dates, values = zip(*valid, strict=True)
                    collected[sname] = pd.Series(values, index=pd.to_datetime(dates), name=sname)
                    logger.debug(f'FRED: {sname} ({sid}): {len(valid)} rows')
                else:
                    logger.debug(f'FRED: {sname} ({sid}): no valid data')
            except Exception as series_err:
                logger.warning(f'FRED: failed {sname} ({sid}): {series_err}')
        return collected

    def _try_load_cache_fallback(self):
        """Try to load cached data as fallback."""
        if self.cache_path.exists():
            try:
                cached = pd.read_parquet(self.cache_path)
                if not cached.empty:
                    logger.info(f'Using cached macro data as fallback ({len(cached)} rows).')
                    return cached
            except Exception as e:
                logger.warning(f'Failed to load macro cache: {e}')
        return pd.DataFrame()

    def _normalize_date_range(self, start_date, end_date):
        """Normalize date range to strings."""
        start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        return start_date_str, end_date_str

    def _strip_timezone_from_index(self, df):
        """Strip timezone from DataFrame index."""
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def _load_macro_data(self, start_date: datetime, end_date: datetime) ->pd.DataFrame:
        if self._is_cache_valid(start_date, end_date):
            logger.info(f'Loading macro data from cache: {self.cache_path}')
            return pd.read_parquet(self.cache_path)

        logger.info('Cache not found or outdated. Loading data from FRED...')
        series_ids = list(self.config.values())
        series_names = list(self.config.keys())

        try:
            start_date_str, end_date_str = self._normalize_date_range(start_date, end_date)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Loading FRED data from {start_date_str} to {end_date_str}')

            session = self._create_fred_session()
            collected = self._load_fred_series(series_ids, series_names, start_date_str, end_date_str, session)

            if not collected:
                logger.warning('FRED API: no series loaded.')
                if self.cache_path.exists():
                    cached = pd.read_parquet(self.cache_path)
                    if not cached.empty:
                        logger.info('Using stale cache as fallback.')
                        return cached
                return pd.DataFrame()

            fred_data = pd.DataFrame(collected)
            fred_data = self._strip_timezone_from_index(fred_data)
            fred_data.to_parquet(self.cache_path)
            logger.info(f'Macro data saved to cache: {self.cache_path} ({len(fred_data)} rows, {len(fred_data.columns)} cols)')
            return fred_data

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Error loading data from FRED: {e}')
            return self._try_load_cache_fallback()

    def _strip_timezone_from_date(self, date):
        """Strip timezone from date if present."""
        if hasattr(date, 'tzinfo') and date.tzinfo is not None:
            return date.replace(tzinfo=None)
        return date

    def _check_cache_full_coverage(self, cache_min, cache_max, start_date, end_date):
        """Check if cache fully covers the required date range."""
        cache_min = pd.Timestamp(cache_min)
        cache_max = pd.Timestamp(cache_max)
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        if cache_min <= start_date and cache_max >= end_date:
            logger.info('Cache fully covers the required date range.')
            return True
        return False

    def _check_cache_partial_coverage(self, cache_max, end_date, cache_min, start_date):
        """Check if cache partially covers the date range with age constraints."""
        cache_age_days = (pd.Timestamp.now() - cache_max).days
        same_month = (cache_max.year == end_date.year and cache_max.month == end_date.month)
        recent_enough = cache_age_days <= 8  # 7 days + 1 buffer

        if same_month and recent_enough and cache_min <= start_date:
            logger.info(f'Cache covers same month (age={cache_age_days}d). Using cached FRED data.')
            return True

        # Also valid if cache covers previous month and current month has no new FRED releases yet
        prev_month_ok = (cache_age_days <= 35 and cache_min <= start_date)
        if prev_month_ok:
            logger.info(f'Using FRED cache (age={cache_age_days}d, max={cache_max.date()}) as valid fallback.')
            return True

        return False

    def _is_cache_valid(self, start_date: datetime, end_date: datetime) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            cached_df = pd.read_parquet(self.cache_path)

            if cached_df.empty:
                logger.warning("Cache exists but is empty. Refresh required.")
                return False

            cache_min = cached_df.index.min()
            cache_max = cached_df.index.max()

            # Strip timezone from cache index if present
            cache_min = self._strip_timezone_from_date(cache_min)
            cache_max = self._strip_timezone_from_date(cache_max)

            # Strip timezone from comparison dates if present
            start_date = self._strip_timezone_from_date(start_date)
            end_date = self._strip_timezone_from_date(end_date)

            # Check full coverage first
            if self._check_cache_full_coverage(cache_min, cache_max, start_date, end_date):
                return True

            # Check partial coverage with age constraints
            if self._check_cache_partial_coverage(cache_max, end_date, cache_min, start_date):
                return True

            logger.info(f'Cache stale (age={(pd.Timestamp.now() - cache_max).days}d, max={cache_max.date()}). Refresh required.')
            return False
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.info(f'Error reading cache file {self.cache_path}: {e}. A reload will be performed.')
            return False
