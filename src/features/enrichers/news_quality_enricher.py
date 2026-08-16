from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("NewsQualityEnricher")

# Константа для уникнення дублювання
DATETIME64_NS = 'datetime64[ns]'

class NewsQualityEnricher(BaseEnricher):
    """
    Enriches DataFrame with news quality metrics:
    - News source diversity
    - News freshness (time since last news)
    - News quality score (based on length, completeness)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with optional config from FeatureOrchestrator."""
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        self.config = config or {}
        logger.info("NewsQualityEnricher initialized")

    @property
    def name(self) -> str:
        return "news_quality"

    @property
    def priority(self) -> int:
        """Run after keyword_entity (35), before sentiment (40)"""
        return 38

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds news quality features to the DataFrame.

        Args:
            df: Input DataFrame with DatetimeIndex
            **kwargs: Should contain 'news' DataFrame

        Returns:
            DataFrame with added news quality features
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping news quality enrichment.")
            return df

        news_df = kwargs.get('news')
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            logger.warning("No news data available in kwargs. Skipping news quality enrichment.")
            return df

        time_col = self._find_time_column(news_df)
        if time_col is None:
            return df

        try:
            news_copy = self._prepare_news_data(news_df, time_col)
            aggregated = self._calculate_quality_metrics(news_copy)
            # ✅ FIX: pass the actual time column, not index (which may be RangeIndex)
            news_timestamps = pd.to_datetime(news_copy[time_col], errors='coerce')
            return self._merge_with_main_dataframe(df, aggregated, news_timestamps)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error during news quality enrichment: {e}")
            return df

    def _find_time_column(self, news_df: pd.DataFrame) -> str | None:
        """Знаходить часову колонку в DataFrame новин."""
        possible_time_cols = ['published_date', 'published_at', 'publishedAt', 'date', 'timestamp', 'datetime', 'time']
        for col in possible_time_cols:
            if col in news_df.columns:
                logger.info(f"✅ Found time column: '{col}'")
                return col

        logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping news quality enrichment.")
        return None

    def _prepare_news_data(self, news_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Підготовлює дані новин: нормалізує час та розраховує completeness."""
        news_copy = news_df.copy()

        # Normalize timezone and convert to datetime64[ns]
        news_copy[time_col] = pd.to_datetime(news_copy[time_col], errors='coerce', utc=True)
        if news_copy[time_col].dt.tz is not None:
            # ✅ FIX: use tz_convert(None) instead of tz_localize(None) to strip UTC tz
            news_copy[time_col] = news_copy[time_col].dt.tz_convert(None)
        news_copy[time_col] = news_copy[time_col].astype(DATETIME64_NS)
        news_copy = news_copy.dropna(subset=[time_col])

        logger.info(f"✅ Found time column '{time_col}' with {len(news_copy)} valid timestamps")

        # Calculate text completeness score (0-1)
        text_cols = ['title', 'description', 'content']
        news_copy['text_completeness'] = 0.0
        for col in text_cols:
            if col in news_copy.columns:
                news_copy['text_completeness'] += news_copy[col].fillna('').str.len().apply(
                    lambda x: min(x / 100, 1.0)
                ) / len(text_cols)

        # Calculate source diversity
        if 'source' in news_copy.columns:
            news_copy['has_source'] = news_copy['source'].notna().astype(int)
        else:
            news_copy['has_source'] = 0

        return news_copy

    def _calculate_quality_metrics(self, news_copy: pd.DataFrame) -> pd.DataFrame:
        """Розраховує та агрегує метрики якості новин."""
        logger.info(f"Calculating news quality metrics for {len(news_copy)} news items...")

        # Set time column as index and aggregate by hour
        # ✅ FIX: index.name is None for RangeIndex — use the time column directly
        if not isinstance(news_copy.index, pd.DatetimeIndex):
            # Find a datetime column and set it as index
            for col in ['published_date', 'published_at', 'datetime', 'timestamp', 'date']:
                if col in news_copy.columns:
                    news_copy = news_copy.set_index(col)
                    break
        # If already DatetimeIndex, use it directly; otherwise resample will fail gracefully

        aggregated = news_copy.resample('1h').agg({
            'text_completeness': 'mean',
            'has_source': 'sum'
        })

        # Rename for clarity
        return aggregated.rename(columns={
            'text_completeness': 'news_quality_score',
            'has_source': 'news_source_count'
        })

    def _merge_with_main_dataframe(self, df: pd.DataFrame, aggregated: pd.DataFrame,
                                  news_timestamps: pd.Series) -> pd.DataFrame:
        """Зливає агреговані дані з основним DataFrame."""
        df_enriched = df.copy()

        # Ensure df has DatetimeIndex
        if not isinstance(df_enriched.index, pd.DatetimeIndex):
            if 'datetime' in df_enriched.columns:
                df_enriched = df_enriched.set_index('datetime')
            else:
                logger.error("Cannot merge: df has no DatetimeIndex or 'datetime' column")
                return df

        # Normalize timezones — use tz_convert(None) for tz-aware, not tz_localize
        if df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_convert(None)
        if aggregated.index.tz is not None:
            aggregated.index = aggregated.index.tz_convert(None)

        # Prepare dataframes for merge_asof
        df_reset = self._normalize_datetime_column(df_enriched.reset_index())

        # ✅ FIX: after reset_index(), the index column may be named differently
        agg_reset = aggregated.reset_index()
        # Rename the first column (which was the index) to 'datetime' if needed
        if 'datetime' not in agg_reset.columns:
            first_col = agg_reset.columns[0]
            agg_reset = agg_reset.rename(columns={first_col: 'datetime'})
        aggregated_reset = self._normalize_datetime_column(agg_reset)

        # Bounded by the bar's own spacing. An unbounded backward asof carries
        # the last aggregate forward for as long as the series runs, so
        # `news_quality_available` below answered "have we ever collected news"
        # rather than "is there news on this bar" -- it equalled news_coverage
        # to four decimal places on all three timeframes (15m 1.0000, 60m
        # 0.2153, 1d 0.2037), which is the era, not the bar.
        df_merged = pd.merge_asof(
            df_reset.sort_values('datetime'),
            aggregated_reset.sort_values('datetime'),
            on='datetime',
            direction='backward',
            tolerance=self.bar_window(df_reset['datetime']),
        )

        df_merged = df_merged.set_index('datetime')

        # "Were there sources in this bar's window", not "did a row match".
        #
        # The aggregation emits a row for every bucket in its span, empty ones
        # included, so a match always exists inside the collected era and
        # `notna()` reproduced that era exactly: 0.2153 on 60m against a
        # news-era fraction of 0.2153, to four decimal places. That made this
        # flag a copy of `news_coverage`, which is computed six lines below and
        # is SUPPOSED to mark the era. Two columns, one fact.
        df_merged['news_quality_available'] = (
            pd.to_numeric(df_merged['news_source_count'], errors='coerce')
            .fillna(0).gt(0).astype(int)
        )
        df_merged['news_quality_score'] = df_merged['news_quality_score'].where(
            df_merged['news_quality_score'].notna(), 0.0)
        df_merged['news_source_count'] = df_merged['news_source_count'].where(
            df_merged['news_source_count'].notna(), 0).astype(int)

        # Calculate news freshness
        df_merged['news_freshness_hours'] = self._calculate_news_freshness(df_merged.index, news_timestamps)

        # Whether this bar could have had news at all, as distinct from
        # happening not to.
        #
        # Every other news flag answers "is there a signal on this bar", and
        # answers 0 both for a quiet hour inside our coverage and for a bar
        # from before we collected anything. Those are not the same fact, and
        # nothing in the frame separated them. collectors.yaml cites exactly
        # this as the reason hourly price history is pinned to 180 days while
        # Yahoo serves 730: extending it would add ~17 months of bars whose
        # 144 news features are zero, teaching the models that the past was
        # newsless.
        #
        # The earliest story we hold is a property of our collection, not of
        # the market, and it is historical — so using it is metadata, not
        # look-ahead. With it, a zero before coverage is labelled as such and
        # price history may safely outrun news history.
        df_merged['news_coverage'] = self._coverage_flag(
            df_merged.index, news_timestamps
        )

        # Calculate metrics using values for robustness
        avg_quality = df_merged['news_quality_score'].values.mean()
        avg_freshness = df_merged['news_freshness_hours'].values.mean()

        logger.info(f"✅ Added news quality features. Avg quality: {avg_quality:.2f}, Avg freshness: {avg_freshness:.1f}h")
        return df_merged

    @staticmethod
    def _coverage_flag(bar_index: pd.DatetimeIndex,
                       news_timestamps: pd.Series) -> pd.Series:
        """1 where the bar falls inside the collected news window, else 0.

        Compared against the EARLIEST story held, not against the nearest one:
        the question is whether we were collecting at all, and a quiet week
        inside the window must stay distinguishable from a year before it.

        Returns 0 everywhere when no usable timestamps exist, which is the
        honest reading — nothing is covered by a collection that produced
        nothing.
        """
        stamps = pd.to_datetime(news_timestamps, errors='coerce').dropna()
        if stamps.empty:
            return pd.Series(0, index=bar_index, dtype=int)

        first = stamps.min()
        bars = pd.Series(bar_index, index=bar_index)
        if getattr(bars.dt, 'tz', None) is not None and first.tzinfo is None:
            first = first.tz_localize(bars.dt.tz)
        elif getattr(bars.dt, 'tz', None) is None and first.tzinfo is not None:
            first = first.tz_localize(None)
        return (bars >= first).astype(int)

    def _normalize_datetime_column(self, df: pd.DataFrame, col_name: str = 'datetime') -> pd.DataFrame:
        """Нормалізує колонку datetime до timezone-naive та datetime64[ns]."""
        if col_name in df.columns:
            df = df.rename(columns={col_name: 'datetime'}) if col_name != 'datetime' else df

            if pd.api.types.is_datetime64_any_dtype(df['datetime']):
                if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
                    # ✅ FIX: tz_convert strips existing tz, tz_localize would raise if already tz-aware
                    df['datetime'] = df['datetime'].dt.tz_convert(None)
                # Convert to ns precision
                if df['datetime'].dtype != DATETIME64_NS:
                    df['datetime'] = df['datetime'].astype(DATETIME64_NS)

        return df

    def _calculate_news_freshness(self, df_index: pd.DatetimeIndex, news_timestamps: pd.Series) -> pd.Series:
        """Розраховує freshness (години з останньої новини) для кожного рядка."""
        freshness = []

        # ✅ FIX: ensure news_timestamps is datetime dtype before using .dt accessor
        try:
            if not pd.api.types.is_datetime64_any_dtype(news_timestamps):
                news_timestamps = pd.to_datetime(news_timestamps, errors='coerce')
            # Normalize timezone
            if hasattr(news_timestamps, 'dt') and hasattr(news_timestamps.dt, 'tz') and news_timestamps.dt.tz is not None:
                news_timestamps = news_timestamps.dt.tz_convert(None)
        except Exception as e:
            self.logger.warning(f"Timestamp conversion failed, attempting fallback: {e}")
            news_timestamps = pd.to_datetime(news_timestamps, errors='coerce')

        for idx in df_index:
            # Normalize timezone для idx якщо потрібно
            idx_normalized = idx
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx_normalized = idx.tz_convert(None)

            # Find most recent news before this timestamp
            recent_news = news_timestamps[news_timestamps <= idx_normalized]
            if not recent_news.empty:
                time_diff = (idx_normalized - recent_news.max()).total_seconds() / 3600
                freshness.append(time_diff)
            else:
                freshness.append(999.0)  # No news available

        return pd.Series(freshness, index=df_index)
