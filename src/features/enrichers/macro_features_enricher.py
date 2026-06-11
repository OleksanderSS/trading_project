from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("MacroFeaturesEnricher")

# Constants to avoid duplication
DATETIME64_NS = "datetime64[ns]"


class MacroFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with macroeconomic indicators from FRED.
    Implements caching to avoid repeated downloads.
    """

    @property
    def name(self) -> str:
        return "macro_features"

    @property
    def priority(self) -> int:
        """Execution order - run after technical analysis (20) but before NLP (30)"""
        return 27

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with optional config dict from FeatureOrchestrator"""
        super().__init__()  # Initialize logger from BaseEnricher
        # ✅ FIX: Correct configuration path
        # First try enrichment.macro_features.macro_fred_series
        config_manager = get_current_config()
        self.config = config_manager.get(
            'enrichment.macro_features.macro_fred_series', {}
        )

        # Fallback: if not found, try macro_features.macro_fred_series
        if not self.config:
            self.config = config_manager.get(
                'macro_features.macro_fred_series', {}
            )

        self.cache_path = Path('./cache') / 'macro_data.parquet'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config:
            logger.warning(
                "Configuration for macro features ('macro_fred_series') not found.")
        else:
            logger.info(
                f"✅ MacroFeaturesEnricher initialized with {len(self.config)} series")

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare DataFrame for enrichment."""
        if df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
                logger.info("Converted 'datetime' column to DatetimeIndex")
            elif 'date' in df.columns:
                df = df.set_index('date')
                logger.info("Converted 'date' column to DatetimeIndex")
            else:
                logger.error(
                    "Cannot enrich macro features: no DatetimeIndex "
                    "or 'datetime' column"
                )
                return pd.DataFrame()  # Return empty DataFrame to skip enrichment

        return df

    def _filter_test_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame for test ticker if in test mode."""
        import json
        config_manager = get_current_config()
        params_path = config_manager.get_runtime_params_path()

        if params_path.exists():
            try:
                with open(params_path) as f:
                    runtime_params = json.load(f)
                test_ticker = runtime_params.get('test_mode', {}).get('test_ticker')
                if test_ticker and 'ticker' in df.columns:
                    logger.info(
                        f"MacroFeaturesEnricher: filtering for ticker {test_ticker}"
                    )
                    df = df[df['ticker'] == test_ticker]
            except Exception as e:
                logger.debug(
                    f"Could not load runtime params for test ticker filtering: {e}"
                )

        return df

    def _prepare_macro_data(self, **kwargs) -> pd.DataFrame:
        """Prepare macro data from Stage 1 or load from FRED API."""
        macro_data = kwargs.get('macro_data')

        if (
            macro_data is not None and
            isinstance(macro_data, pd.DataFrame) and
            not macro_data.empty
        ):
            logger.info(
                f"Using macro_data from Stage 1 "
                f"({len(macro_data)} rows)"
            )
            return self._pivot_macro_data(macro_data)
        else:
            logger.info("No macro_data in kwargs, loading from FRED API...")
            # ✅ FIX: Check if index is datetime, otherwise use datetime column
            if isinstance(self.df.index, pd.DatetimeIndex):
                start_date = self.df.index.min()
                end_date = self.df.index.max()
            elif 'datetime' in self.df.columns:
                start_date = pd.to_datetime(self.df['datetime']).min()
                end_date = pd.to_datetime(self.df['datetime']).max()
            else:
                # Fallback to reasonable defaults
                logger.warning("No datetime index or column found, using default date range")
                start_date = pd.Timestamp('2020-01-01')
                end_date = pd.Timestamp.now()

            return self._load_macro_data(start_date, end_date)

    def _pivot_macro_data(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Pivot macro data from long to wide format."""
        if 'series_id' not in macro_data.columns or 'value' not in macro_data.columns:
            return macro_data

        # Find date column
        date_col = None
        for col in ['date', 'datetime', 'realtime_start']:
            if col in macro_data.columns:
                date_col = col
                break

        if not date_col:
            logger.warning("No date column found in macro_data for pivoting")
            return macro_data

        macro_data[date_col] = pd.to_datetime(macro_data[date_col])
        macro_pivoted = macro_data.pivot_table(
            index=date_col,
            columns='series_id',
            values='value',
            aggfunc='last'
        )
        macro_pivoted.columns = [f'FRED_{col}' for col in macro_pivoted.columns]
        logger.info(
            f"Pivoted macro data into {len(macro_pivoted.columns)} FRED columns"
        )
        return macro_pivoted

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

        # Store df for use in helper methods
        self.df = df

        # Validate and prepare DataFrame
        df = self._validate_dataframe(df)
        if df.empty:
            return df

        # Filter for test ticker if needed
        df = self._filter_test_ticker(df)

        # Log processing info
        start_date = df.index.min()
        end_date = df.index.max()
        unique_dates = len(df.index.unique())
        logger.info(
            f"MacroFeaturesEnricher processing {len(df)} records "
            f"({unique_dates} unique dates) from {start_date} to {end_date}"
        )

        # Prepare macro data
        macro_data = self._prepare_macro_data(**kwargs)

        if macro_data.empty:
            logger.warning("Could not load macro data. Skipping enrichment.")
            return df

        return self._merge_macro_data(df, macro_data)

    def _prepare_macro_index(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare macro data index for merging."""
        # Remove index duplicates for correct reindex
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            if 'datetime' in macro_data.columns:
                macro_data = macro_data.set_index('datetime')
            elif 'date' in macro_data.columns:
                macro_data = macro_data.set_index('date')

        # Ensure macro_data is sorted and deduplicated
        macro_data = macro_data.sort_index()
        macro_data = macro_data[~macro_data.index.duplicated(keep='last')]

        # Normalize timezone/precision in macro_data.index
        if isinstance(macro_data.index, pd.DatetimeIndex):
            if macro_data.index.tz is not None:
                macro_data.index = macro_data.index.tz_localize(None)
            if macro_data.index.dtype != DATETIME64_NS:
                macro_data.index = macro_data.index.astype(DATETIME64_NS)

        return macro_data

    def _normalize_datetime_column(
        self, df: pd.DataFrame, datetime_col: str = 'datetime'
    ) -> pd.DataFrame:
        """Normalize datetime column timezone and precision."""
        if datetime_col not in df.columns:
            return df

        if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            # Remove timezone if present
            if (
                hasattr(df[datetime_col].dtype, 'tz') and
                df[datetime_col].dt.tz is not None
            ):
                df[datetime_col] = df[datetime_col].dt.tz_localize(None)
            # Convert to ns precision
            if df[datetime_col].dtype != DATETIME64_NS:
                df[datetime_col] = df[datetime_col].astype(DATETIME64_NS)

        return df

    def _merge_with_duplicates(
        self, df: pd.DataFrame, macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge data when DataFrame has duplicate index labels (multiple tickers)."""
        logger.info(
            "Detected duplicate index labels (multiple tickers). "
            "Using merge instead of reindex."
        )

        # Reset indices
        df_reset = df.reset_index()
        macro_reset = macro_data.reset_index()

        # Find and normalize datetime column in df_reset
        datetime_col = None
        for col in ['datetime', 'index', 'date']:
            if col in df_reset.columns:
                datetime_col = col
                break

        if datetime_col is None:
            logger.error(
                "Cannot find datetime column after reset_index"
            )
            return df

        if datetime_col != 'datetime':
            df_reset = df_reset.rename(columns={datetime_col: 'datetime'})

        df_reset = self._normalize_datetime_column(df_reset, 'datetime')

        # Find and normalize datetime column in macro_reset
        macro_datetime_col = None
        for col in ['datetime', 'index', 'date']:
            if col in macro_reset.columns:
                macro_datetime_col = col
                break

        if macro_datetime_col and macro_datetime_col != 'datetime':
            macro_reset = macro_reset.rename(columns={macro_datetime_col: 'datetime'})

        macro_reset = self._normalize_datetime_column(macro_reset, 'datetime')

        # Merge with backward fill
        df_merged = pd.merge_asof(
            df_reset.sort_values('datetime'),
            macro_reset.sort_values('datetime'),
            on='datetime',
            direction='backward'
        )

        # Restore index
        if 'datetime' in df_merged.columns:
            return df_merged.set_index('datetime')
        else:
            logger.error("'datetime' column missing after merge_asof")
            return df_merged

    def _merge_without_duplicates(
        self, df: pd.DataFrame, macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge data when DataFrame has unique index labels."""
        # Normalize datetime columns to avoid timezone conflicts
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df_index = df.index.tz_localize(None)
        else:
            df_index = df.index

        if hasattr(macro_data.index, 'tz') and macro_data.index.tz is not None:
            macro_data_index = macro_data.index.tz_localize(None)
        else:
            macro_data_index = macro_data.index

        macro_dates = macro_data_index.unique()
        aligned_macro_data = macro_data.loc[macro_dates].reindex(
            df_index, method='ffill'
        )

        # Combine without Join to avoid duplicating rows on timestamp duplicates
        for col in aligned_macro_data.columns:
            if col not in df.columns:  # Do not overwrite existing columns
                df[col] = aligned_macro_data[col].values

        return df

    def _post_process_fred_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process FRED columns with forward fill and NaN handling."""
        # Fix: Handle None values in columns
        fred_cols = [col for col in df.columns if col and isinstance(col, str) and col.startswith('FRED_')]
        if not fred_cols:
            return df

        # Convert to numeric before operations
        for col in fred_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Forward fill up to 60 days (2 months)
        df[fred_cols] = df[fred_cols].ffill(limit=60)

        # Handle remaining NaN values
        remaining_nans = df[fred_cols].isna().sum()
        if remaining_nans.any():
            logger.warning(
                f"Some FRED columns still have NaN after ffill: "
                f"{remaining_nans[remaining_nans > 0].to_dict()}"
            )
            # Fill remaining NaN with median
            df[fred_cols] = df[fred_cols].fillna(df[fred_cols].median())

        return df

    def _merge_macro_data(
        self, df: pd.DataFrame, macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge macro data with main DataFrame."""
        logger.info("Joining macro data with the main DataFrame...")

        # Prepare macro data index
        macro_data = self._prepare_macro_index(macro_data)

        # Choose merge strategy based on index duplicates
        if df.index.duplicated().any():
            df = self._merge_with_duplicates(df, macro_data)
        else:
            df = self._merge_without_duplicates(df, macro_data)

        # Backward fill
        df = df.bfill()

        # Post-process FRED columns
        df = self._post_process_fred_columns(df)

        logger.info(
            f"Macro features successfully added. Final shape: {df.shape}"
        )
        return df

    def _load_macro_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        if self._is_cache_valid(start_date, end_date):
            logger.info(
                f"Loading macro data from cache: "
                f"{self.cache_path}"
            )
            return pd.read_parquet(self.cache_path)

        logger.info("Cache not found or outdated. Loading data from FRED...")
        series_ids = list(self.config.values())
        series_names = list(self.config.keys())

        try:
            # ✅ FIX: Load FRED API key from environment
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            fred_api_key = os.getenv('FRED_API_KEY')
            if not fred_api_key:
                logger.warning("FRED_API_KEY not found in environment. Macro data loading may fail.")
            else:
                logger.debug(f"Using FRED API key: {fred_api_key[:8]}...")
            
            # ✅ FIX: Convert Timestamp to string for web.DataReader
            start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

            logger.debug(f"Loading FRED data from {start_date_str} to {end_date_str}")
            # ✅ FIX: Add timeout and retry logic
            import pandas_datareader.data as web
            import requests

            # Set session with longer timeout and retry logic
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            class TimeoutSession(requests.Session):
                """Session with default timeout and retry logic."""
                def __init__(self, timeout=300, retries=3, backoff_factor=1):
                    super().__init__()
                    self.timeout = timeout
                    retry_strategy = Retry(
                        total=retries,
                        backoff_factor=backoff_factor,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["HEAD", "GET", "OPTIONS"]
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    self.mount("https://", adapter)
                    self.mount("http://", adapter)

                def request(self, method, url, **kwargs):
                    kwargs.setdefault('timeout', self.timeout)
                    return super().request(method, url, **kwargs)

            session = TimeoutSession(timeout=300, retries=3, backoff_factor=1)

            fred_data = web.DataReader(series_ids, 'fred', start_date_str, end_date_str, api_key=fred_api_key)
            
            # ✅ FIX: Check if data is empty
            if fred_data.empty:
                logger.warning("FRED API returned empty data. This may be due to missing API key or network issues.")
                return pd.DataFrame()
            
            fred_data.columns = series_names

            fred_data.to_parquet(self.cache_path)
            logger.info(
                f"Macro data saved to cache: {self.cache_path}"
            )
            return fred_data
        except Exception as e:
            logger.error(
                f"Error loading data from FRED: {e}"
            )
            # ✅ FALLBACK: If FRED loading fails, try to use whatever we have in cache
            if self.cache_path.exists():
                logger.info("Falling back to existing cache despite loading error...")
                try:
                    return pd.read_parquet(self.cache_path)
                except Exception:
                    pass
            return pd.DataFrame()

    def _is_cache_valid(self, start_date: datetime, end_date: datetime) -> bool:
        if not self.cache_path.exists():
            return False

        try:
            cached_df = pd.read_parquet(self.cache_path)
            
            # ✅ FIX: Check if cache is empty
            if cached_df.empty:
                logger.warning("Cache exists but is empty. Refresh required.")
                return False
            
            if (
                cached_df.index.min() <= start_date and
                cached_df.index.max() >= end_date
            ):
                logger.info("Cache fully covers the required date range.")
                return True
            else:
                logger.info(
                    "Date range in cache is insufficient. Refresh required."
                )
                return False
        except Exception as e:
            logger.info(
                f"Error reading cache file {self.cache_path}: {e}. "
                "A reload will be performed."
            )
            return False
