from typing import Any

import numpy as np
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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
        news_count = self._normalize_news_count_datetime(news_count, time_col)

        return self._merge_hype_scores(df_enriched, news_count, ['ticker', 'datetime'], "per ticker")

    def _calculate_global_hype(self, df_enriched: pd.DataFrame, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Calculate global hype scores."""
        news_count = self._aggregate_news_globally(news_copy, time_col)
        news_count = self._normalize_news_count_datetime(news_count, time_col)

        return self._merge_hype_scores(df_enriched, news_count, ['datetime'], "global")

    def _merge_hype_scores(self, df_enriched: pd.DataFrame, news_count: pd.DataFrame, merge_keys: list, hype_type: str) -> pd.DataFrame:
        """Merge hype scores into DataFrame and add hype_score column."""
        # Earlier enrichers set `datetime` as the index, and this returned
        # the frame untouched when it was not a column -- silently, adding
        # nothing, which is half of why this enricher reported +0 columns.
        # Take it from wherever it is and put it back the same way.
        restore_index = False
        if 'datetime' not in df_enriched.columns:
            if isinstance(df_enriched.index, pd.DatetimeIndex):
                df_enriched = df_enriched.reset_index()
                if 'index' in df_enriched.columns and 'datetime' not in df_enriched.columns:
                    df_enriched = df_enriched.rename(columns={'index': 'datetime'})
                restore_index = True
                # The tz strip in _normalize_dataframe_datetime only runs on
                # a 'datetime' COLUMN, so an index kept its tz and merge_asof
                # refused to join tz-aware bars to tz-naive news windows --
                # raising into the caller's except, which returned the frame
                # untouched. Same silence, different cause.
                self._normalize_dataframe_datetime(df_enriched)
            else:
                logger.error(
                    "No 'datetime' column or DatetimeIndex; hype scores "
                    "cannot be attached to these bars."
                )
                return df_enriched

        # merge_asof, not an exact match: a bar takes the most recent closed
        # window, so 14:15 and 14:30 inherit the 13:00-13:59 count instead of
        # nothing at all. An exact merge on the hour only ever matched bars
        # whose clock reading equalled a window label.
        if 'datetime' not in news_count.columns:
            logger.error(
                "News counts have no 'datetime' after normalisation (have "
                "%s); hype scores not attached.", list(news_count.columns),
            )
            return df_enriched
        right = news_count.dropna(subset=['datetime']).sort_values('datetime')
        if 'ticker' in merge_keys and 'ticker' in right.columns:
            parts = []
            for ticker, group in df_enriched.groupby('ticker', sort=False):
                per = right[right['ticker'] == ticker].drop(columns=['ticker'])
                if per.empty:
                    parts.append(group.assign(news_count=np.nan))
                    continue
                # Keep each row's original label: merge_asof returns a
                # fresh RangeIndex, and concatenating per-ticker pieces then
                # yields duplicates, which breaks any downstream reindex.
                labelled = group.sort_values('datetime').copy()
                labelled['__row_label'] = labelled.index
                merged = pd.merge_asof(
                    labelled,
                    per.drop_duplicates(subset=['datetime'], keep='last'),
                    on='datetime', direction='backward',
                )
                merged.index = merged['__row_label'].to_numpy()
                parts.append(merged.drop(columns=['__row_label']))
            df_enriched = pd.concat(parts).sort_index()
        else:
            df_enriched = pd.merge_asof(
                df_enriched.sort_values('datetime'),
                right.drop(columns=['ticker'], errors='ignore')
                     .drop_duplicates(subset=['datetime'], keep='last'),
                on='datetime', direction='backward',
            )
        df_enriched['hype_available'] = df_enriched['news_count'].notna().astype(int)
        df_enriched['hype_score'] = df_enriched['news_count'].where(df_enriched['news_count'].notna(), 0)
        df_enriched = df_enriched.drop(columns=['news_count'])
        if restore_index:
            df_enriched = df_enriched.set_index('datetime')
        logger.info(f"Added {hype_type} hype_score based on news count")

        return df_enriched

    def _aggregate_news_by_ticker(self, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Aggregate news count by ticker and time."""
        return news_copy.groupby(['ticker', pd.Grouper(key=time_col, freq='1h')]).size().reset_index(name='news_count')

    def _aggregate_news_globally(self, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Aggregate news count globally by time."""
        return news_copy.groupby(pd.Grouper(key=time_col, freq='1h')).size().reset_index(name='news_count')

    def _normalize_news_count_datetime(
        self, news_count: pd.DataFrame, time_col: str | None = None
    ) -> pd.DataFrame:
        """Name the time column 'datetime' and mark it as a window END.

        The guard here read `if 'datetime' not in news_count.columns: return`
        -- returning early in exactly the case the rename existed to handle.
        The aggregations produce [ticker, published_at, news_count], so
        'datetime' is never present, the function always returned untouched,
        and the merge that followed asked for a column that did not exist.
        Every run logged

            Enricher 'hype_features' completed: +0 columns in 0.17s

        after counting all 15,274 articles. Reproduced directly: zero columns
        added, whether datetime arrives as a column or as the index.

        The rename was wrong twice over: `columns[0]` is 'ticker' in the
        per-ticker aggregation, so had the guard let it run it would have
        renamed the ticker.

        The +1h is the same correction applied to the sentiment and keyword
        windows: `pd.Grouper(freq='1h')` labels a bucket with its start, so
        articles published up to 14:59 sit under 14:00 and would reach a bar
        at 14:00 that could not have read them.
        """
        if 'datetime' not in news_count.columns:
            source = time_col if time_col in news_count.columns else None
            if source is None:
                source = next(
                    (c for c in news_count.columns
                     if pd.api.types.is_datetime64_any_dtype(news_count[c])),
                    None,
                )
            if source is None:
                logger.error(
                    "News counts carry no time column (have %s); hype scores "
                    "cannot be attached.", list(news_count.columns),
                )
                return news_count
            news_count = news_count.rename(columns={source: 'datetime'})

        if pd.api.types.is_datetime64_any_dtype(news_count['datetime']):
            if hasattr(news_count['datetime'].dtype, 'tz') and news_count['datetime'].dt.tz is not None:
                news_count['datetime'] = news_count['datetime'].dt.tz_localize(None)

            if news_count['datetime'].dtype != DATETIME64_NS:
                news_count['datetime'] = news_count['datetime'].astype(DATETIME64_NS)

            news_count['datetime'] = news_count['datetime'] + pd.Timedelta(hours=1)

        return news_count
