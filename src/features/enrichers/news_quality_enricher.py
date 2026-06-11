from typing import Any

import numpy as np
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
        self._analysis_cache: dict[str, Any] = {}
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
            # ✅ OPTIMIZATION: Cache analysis results based on news_df hash
            import hashlib
            news_hash = hashlib.sha256(pd.util.hash_pandas_object(news_df, index=True).values.tobytes()).hexdigest()

            if news_hash in self._analysis_cache:
                logger.info("🚀 Using cached news quality analysis results")
                aggregated, news_timestamps = self._analysis_cache[news_hash]
            else:
                logger.info(f"🔄 Performing fresh news quality analysis for {len(news_df)} items")
                news_copy = self._prepare_news_data(news_df, time_col)
                aggregated = self._calculate_quality_metrics(news_copy)
                # Create proper Series with datetime values
                news_timestamps = pd.Series(news_copy.index, dtype='datetime64[ns]')
                self._analysis_cache[news_hash] = (aggregated, news_timestamps)

            return self._merge_with_main_dataframe(df, aggregated, news_timestamps)

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

        # Add news type classification
        news_copy['type'] = news_copy.apply(self._classify_news_type, axis=1)

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

    def _classify_news_type(self, row: pd.Series) -> str:
        """Classify news type based on title and content."""
        title = str(row.get('title', '')).lower()
        content = str(row.get('content', '')).lower()
        text = f"{title} {content}"

        # Simple keyword-based classification
        if any(keyword in text for keyword in ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'annual']):
            return 'earnings'
        elif any(keyword in text for keyword in ['merger', 'acquisition', 'buyout', 'takeover', 'deal']):
            return 'merger'
        elif any(keyword in text for keyword in ['sec', 'filing', 'regulation', 'compliance', 'investigation']):
            return 'regulatory'
        elif any(keyword in text for keyword in ['launch', 'product', 'release', 'announcement', 'unveil']):
            return 'product'
        elif any(keyword in text for keyword in ['analyst', 'rating', 'upgrade', 'downgrade', 'target']):
            return 'analyst'
        elif any(keyword in text for keyword in ['market', 'index', 'stocks', 'trading', 'investors']):
            return 'market'
        else:
            return 'general'

    def _calculate_quality_metrics(self, news_copy: pd.DataFrame) -> pd.DataFrame:
        """Розраховує та агрегує метрики якості новин."""
        logger.info(f"Calculating news quality metrics for {len(news_copy)} news items...")

        # Set time column as index and aggregate by hour
        time_col = news_copy.index.name or (news_copy.index.names[0] if news_copy.index.names and news_copy.index.names[0] else 'datetime')
        if time_col and time_col in news_copy.columns:
            news_copy_indexed = news_copy.set_index(time_col)
        else:
            news_copy_indexed = news_copy

        # ✅ FIX: Ensure DatetimeIndex for resampling
        if not isinstance(news_copy_indexed.index, pd.DatetimeIndex):
            news_copy_indexed.index = pd.to_datetime(news_copy_indexed.index)

        aggregated = news_copy_indexed.resample('1h').agg({
            'text_completeness': 'mean',
            'has_source': 'sum'
        })

        # Rename for clarity
        result = aggregated.rename(columns={
            'text_completeness': 'news_quality_score',
            'has_source': 'news_source_count'
        })

        return result

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
        if isinstance(df_enriched.index, pd.DatetimeIndex) and df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_localize(None)
        if isinstance(aggregated.index, pd.DatetimeIndex) and aggregated.index.tz is not None:
            aggregated.index = aggregated.index.tz_localize(None)

        # Prepare dataframes for merge_asof
        df_reset = self._normalize_datetime_column(df_enriched.reset_index())
        aggregated_reset = self._normalize_datetime_column(aggregated.reset_index(), 'index')

        # Ensure both dataframes have ns precision datetime
        if 'datetime' in df_reset.columns and df_reset['datetime'].dtype != DATETIME64_NS:
            df_reset['datetime'] = df_reset['datetime'].astype(DATETIME64_NS)
        if 'datetime' in aggregated_reset.columns and aggregated_reset['datetime'].dtype != DATETIME64_NS:
            aggregated_reset['datetime'] = aggregated_reset['datetime'].astype(DATETIME64_NS)

        # Merge using merge_asof
        df_merged = pd.merge_asof(
            df_reset.sort_values('datetime'),
            aggregated_reset.sort_values('datetime'),
            on='datetime',
            direction='backward'
        )

        df_merged = df_merged.set_index('datetime')

        # Fill missing values
        df_merged['news_quality_score'] = df_merged['news_quality_score'].fillna(0.0)
        df_merged['news_source_count'] = df_merged['news_source_count'].fillna(0).astype(int)

        # Calculate news freshness
        df_merged['news_freshness_hours'] = self._calculate_news_freshness(df_merged.index, news_timestamps)

        logger.info(f"✅ Added news quality features. Avg quality: {df_merged['news_quality_score'].mean():.2f}, Avg freshness: {df_merged['news_freshness_hours'].mean():.1f}h")
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

        # DEBUG: Log the structure of news_timestamps
        logger.debug(f"DEBUG: news_timestamps type: {type(news_timestamps)}")
        logger.debug(f"DEBUG: news_timestamps shape: {getattr(news_timestamps, 'shape', 'no shape')}")
        if len(news_timestamps) > 0:
            logger.debug(f"DEBUG: first element type: {type(news_timestamps.iloc[0])}")
            logger.debug(f"DEBUG: first element value: {news_timestamps.iloc[0]}")

        # Normalize timezone для news_timestamps - перевіряємо чи це DatetimeIndex
        if hasattr(news_timestamps, 'dt') and news_timestamps.dt.tz is not None:
            news_timestamps = news_timestamps.dt.tz_localize(None)

        for idx in df_index:
            # Normalize timezone для idx якщо потрібно
            idx_normalized = idx
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx_normalized = idx.tz_localize(None)

            # Find most recent news before this timestamp
            # Convert to pandas Series for proper comparison
            if not isinstance(news_timestamps, pd.Series):
                news_timestamps = pd.Series(news_timestamps)

            # Handle case where Series contains numpy arrays
            if len(news_timestamps) > 0 and isinstance(news_timestamps.iloc[0], np.ndarray):
                # Flatten numpy arrays to timestamps
                flat_timestamps = []
                for item in news_timestamps:
                    if isinstance(item, np.ndarray):
                        flat_timestamps.extend(item.tolist())
                    else:
                        flat_timestamps.append(item)
                news_timestamps = pd.Series(flat_timestamps)

            recent_news = news_timestamps[news_timestamps <= idx_normalized]
            if not recent_news.empty:
                time_diff = (idx_normalized - recent_news.max()).total_seconds() / 3600
                freshness.append(time_diff)
            else:
                freshness.append(999.0)  # No news available

        return pd.Series(freshness, index=df_index)
