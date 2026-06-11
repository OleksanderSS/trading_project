from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("HypeEnricher")

# Constants to avoid duplication
DATETIME64_NS = "datetime64[ns]"
POSSIBLE_TIME_COLS = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']

class HypeEnricher(BaseEnricher):
    """
    Enriches the DataFrame with hype scores by counting news occurrences
    within a rolling time window to gauge market attention.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with optional config from FeatureOrchestrator."""
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        self.config = config or {}

    @property
    def name(self) -> str:
        """Unique identifier for the enricher."""
        return "hype_features"

    @property
    def priority(self) -> int:
        """Execution order - run after NLP (30) and sentiment (40)"""
        return 50

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates hype scores using news data from kwargs.

        Args:
            df: Input DataFrame (market data)
            **kwargs:
                news: DataFrame with news data
                hype_window (str): The rolling window size (e.g., '1h', '24h'). Defaults to '1h'.

        Returns:
            DataFrame with an additional 'hype_score' column.
        """
        if not self._validate_input(df):
            return df

        news_df = kwargs.get('news')
        if not self._validate_news_data(news_df):
            return df

        time_col = self._find_time_column(news_df)
        if time_col is None:
            return df

        hype_window = kwargs.get('hype_window', '1h')
        news_count = len(news_df) if news_df is not None and isinstance(news_df, pd.DataFrame) else 0
        logger.info(f"Calculating hype scores using window: {hype_window} from {news_count} news items")

        try:
            return self._process_hype_enrichment(df, news_df, time_col)
        except Exception as e:
            logger.error(f"Error during hype enrichment: {e}", exc_info=True)
            return df

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame."""
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping hype calculation.")
            return False
        return True

    def _validate_news_data(self, news_df: pd.DataFrame) -> bool:
        """Validate news data."""
        if news_df is None:
            logger.warning("No news data available in kwargs. Skipping hype enrichment.")
            return False

        if not isinstance(news_df, pd.DataFrame):
            logger.warning("News data is not a DataFrame. Skipping hype enrichment.")
            return False

        if news_df.empty:
            logger.warning("News DataFrame is empty. Skipping hype enrichment.")
            return False

        return True

    def _find_time_column(self, news_df: pd.DataFrame) -> str | None:
        """Find time column in news DataFrame."""
        for col in POSSIBLE_TIME_COLS:
            if col in news_df.columns:
                logger.info(f"✅ Found time column '{col}' with {len(news_df)} valid timestamps")
                return col

        logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping hype enrichment.")
        return None

    def _process_hype_enrichment(self, df: pd.DataFrame, news_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Process hype enrichment."""
        df_enriched = df.copy()
        news_copy = self._prepare_news_data(news_df, time_col)

        self._normalize_dataframe_datetime(df_enriched)

        if self._has_ticker_data(news_copy, df_enriched):
            return self._calculate_ticker_hype(df_enriched, news_copy, time_col)
        else:
            return self._calculate_global_hype(df_enriched, news_copy, time_col)

    def _prepare_news_data(self, news_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Prepare news data with normalized datetime."""
        news_copy = news_df.copy()

        # Normalize timezone and convert to datetime64[ns]
        news_copy[time_col] = pd.to_datetime(news_copy[time_col], errors='coerce', utc=True)
        if news_copy[time_col].dt.tz is not None:
            news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
        news_copy[time_col] = news_copy[time_col].astype(DATETIME64_NS)

        return news_copy.dropna(subset=[time_col]).sort_values(time_col)

    def _normalize_dataframe_datetime(self, df: pd.DataFrame):
        """Normalize datetime column in DataFrame."""
        if 'datetime' not in df.columns:
            return

        if pd.api.types.is_datetime64_any_dtype(df['datetime']):
            if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_localize(None)

            if df['datetime'].dtype != DATETIME64_NS:
                df['datetime'] = df['datetime'].astype(DATETIME64_NS)

    def _has_ticker_data(self, news_copy: pd.DataFrame, df_enriched: pd.DataFrame) -> bool:
        """Check if both DataFrames have ticker column."""
        return 'ticker' in news_copy.columns and 'ticker' in df_enriched.columns

    def _calculate_ticker_hype(self, df_enriched: pd.DataFrame, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Calculate hype scores by ticker."""
        news_count = self._aggregate_news_by_ticker(news_copy, time_col)
        news_count = self._normalize_news_count_datetime(news_count)

        return self._merge_hype_scores(df_enriched, news_count, ['ticker', 'datetime'], "per ticker")

    def _calculate_global_hype(self, df_enriched: pd.DataFrame, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Calculate global hype scores."""
        news_count = self._aggregate_news_globally(news_copy, time_col)
        news_count = self._normalize_news_count_datetime(news_count)

        return self._merge_hype_scores(df_enriched, news_count, ['datetime'], "global")

    def _merge_hype_scores(self, df_enriched: pd.DataFrame, news_count: pd.DataFrame, merge_keys: list, hype_type: str) -> pd.DataFrame:
        """Merge hype scores into DataFrame and add hype_score column."""
        if 'datetime' not in df_enriched.columns:
            return df_enriched

        df_enriched = df_enriched.merge(news_count, on=merge_keys, how='left')
        df_enriched['hype_score'] = df_enriched['news_count'].fillna(0)
        df_enriched = df_enriched.drop(columns=['news_count'])
        logger.info(f"Added {hype_type} hype_score based on news count")

        return df_enriched

    def _aggregate_news_by_ticker(self, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Aggregate news count by ticker and time."""
        # Use 'type' column instead of 'ticker' for proper classification
        if 'type' in news_copy.columns:
            return news_copy.groupby(['type', pd.Grouper(key=time_col, freq='1h')]).size().reset_index(name='news_count')
        else:
            return news_copy.groupby(['ticker', pd.Grouper(key=time_col, freq='1h')]).size().reset_index(name='news_count')

    def _aggregate_news_globally(self, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Aggregate news count globally by time."""
        return news_copy.groupby(pd.Grouper(key=time_col, freq='1h')).size().reset_index(name='news_count')

    def _normalize_news_count_datetime(self, news_count: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime in news count DataFrame."""
        if 'datetime' not in news_count.columns:
            return news_count

        news_count = news_count.rename(columns={news_count.columns[0]: 'datetime'})

        if pd.api.types.is_datetime64_any_dtype(news_count['datetime']):
            if hasattr(news_count['datetime'].dtype, 'tz') and news_count['datetime'].dt.tz is not None:
                news_count['datetime'] = news_count['datetime'].dt.tz_localize(None)

            if news_count['datetime'].dtype != DATETIME64_NS:
                news_count['datetime'] = news_count['datetime'].astype(DATETIME64_NS)

        return news_count
