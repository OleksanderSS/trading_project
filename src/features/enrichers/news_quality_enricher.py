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
            return self._merge_with_main_dataframe(df, aggregated, news_copy.index.to_series())

        except Exception as e:
            logger.error(f"Error during news quality enrichment: {e}", exc_info=True)
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
            news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
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
        time_col = news_copy.index.name or news_copy.index.names[0]
        news_copy = news_copy.set_index(time_col)

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

        # Normalize timezones
        if df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_localize(None)
        if aggregated.index.tz is not None:
            aggregated.index = aggregated.index.tz_localize(None)

        # Prepare dataframes for merge_asof
        df_reset = self._normalize_datetime_column(df_enriched.reset_index())
        aggregated_reset = self._normalize_datetime_column(aggregated.reset_index(), 'index')

        # Merge using merge_asof
        df_merged = pd.merge_asof(
            df_reset.sort_values('datetime'),
            aggregated_reset.sort_values('datetime'),
            on='datetime',
            direction='backward'
        )

        df_merged = df_merged.set_index('datetime')

        # Explicit availability flag: missing quality means no aligned news signal.
        df_merged['news_quality_available'] = (
            df_merged[['news_quality_score', 'news_source_count']].notna().any(axis=1).astype(int)
        )
        df_merged['news_quality_score'] = df_merged['news_quality_score'].where(
            df_merged['news_quality_score'].notna(), 0.0)
        df_merged['news_source_count'] = df_merged['news_source_count'].where(
            df_merged['news_source_count'].notna(), 0).astype(int)

        # Calculate news freshness
        df_merged['news_freshness_hours'] = self._calculate_news_freshness(df_merged.index, news_timestamps)

        # Calculate metrics using values for robustness
        avg_quality = df_merged['news_quality_score'].values.mean()
        avg_freshness = df_merged['news_freshness_hours'].values.mean()

        logger.info(f"✅ Added news quality features. Avg quality: {avg_quality:.2f}, Avg freshness: {avg_freshness:.1f}h")
        return df_merged

    def _normalize_datetime_column(self, df: pd.DataFrame, col_name: str = 'datetime') -> pd.DataFrame:
        """Нормалізує колонку datetime до timezone-naive та datetime64[ns]."""
        if col_name in df.columns:
            df = df.rename(columns={col_name: 'datetime'}) if col_name != 'datetime' else df

            if pd.api.types.is_datetime64_any_dtype(df['datetime']):
                if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
                    df['datetime'] = df['datetime'].dt.tz_localize(None)
                # Convert to ns precision
                if df['datetime'].dtype != DATETIME64_NS:
                    df['datetime'] = df['datetime'].astype(DATETIME64_NS)

        return df

    def _calculate_news_freshness(self, df_index: pd.DatetimeIndex, news_timestamps: pd.Series) -> pd.Series:
        """Розраховує freshness (години з останньої новини) для кожного рядка."""
        freshness = []

        # Normalize timezone для news_timestamps
        if news_timestamps.dt.tz is not None:
            news_timestamps = news_timestamps.dt.tz_localize(None)

        for idx in df_index:
            # Normalize timezone для idx якщо потрібно
            idx_normalized = idx
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx_normalized = idx.tz_localize(None)

            # Find most recent news before this timestamp
            recent_news = news_timestamps[news_timestamps <= idx_normalized]
            if not recent_news.empty:
                time_diff = (idx_normalized - recent_news.max()).total_seconds() / 3600
                freshness.append(time_diff)
            else:
                freshness.append(999.0)  # No news available

        return pd.Series(freshness, index=df_index)
