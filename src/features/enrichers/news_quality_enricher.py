import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.features.enrichers.base import BaseEnricher
from src.features.nlp.processors.news_harmonizer import harmonize_batch
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("NewsQualityEnricher")

class NewsQualityEnricher(BaseEnricher):
    """
    Enriches DataFrame with news quality metrics:
    - News source diversity
    - News freshness (time since last news)
    - News quality score (based on length, completeness)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config from FeatureOrchestrator."""
        self.config = config or {}
        logger.info("NewsQualityEnricher initialized")

    @property
    def name(self) -> str:
        return "news_quality"

    @property
    def priority(self) -> int:
        """Run after keyword_entity (35), before sentiment (40)"""
        return 38

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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

        # Get news data from kwargs
        news_df = kwargs.get('news')
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            logger.warning("No news data available in kwargs. Skipping news quality enrichment.")
            return df

        # Find time column
        time_col = None
        possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
        for col in possible_time_cols:
            if col in news_df.columns:
                time_col = col
                break

        if time_col is None:
            logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping news quality enrichment.")
            return df

        try:
            news_copy = news_df.copy()
            # ✅ FIX: Normalize timezone and convert to datetime64[ns]
            news_copy[time_col] = pd.to_datetime(news_copy[time_col], errors='coerce', utc=True)
            if news_copy[time_col].dt.tz is not None:
                news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
            news_copy[time_col] = news_copy[time_col].astype('datetime64[ns]')
            news_copy = news_copy.dropna(subset=[time_col])
            
            logger.info(f"✅ Found time column '{time_col}' with {len(news_copy)} valid timestamps")

            logger.info(f"Calculating news quality metrics for {len(news_copy)} news items...")

            # Calculate quality metrics per news item
            
            # 1. Text completeness score (0-1)
            text_cols = ['title', 'description', 'content']
            news_copy['text_completeness'] = 0.0
            for col in text_cols:
                if col in news_copy.columns:
                    news_copy['text_completeness'] += news_copy[col].fillna('').str.len().apply(
                        lambda x: min(x / 100, 1.0)  # Normalize to 0-1, cap at 100 chars
                    ) / len(text_cols)

            # 2. Source diversity (unique sources per time window)
            if 'source' in news_copy.columns:
                news_copy['has_source'] = news_copy['source'].notna().astype(int)
            else:
                news_copy['has_source'] = 0

            # Aggregate by time (hourly)
            news_copy = news_copy.set_index(time_col)
            
            aggregated = news_copy.resample('1h').agg({
                'text_completeness': 'mean',
                'has_source': 'sum'
            })

            # Rename for clarity
            aggregated = aggregated.rename(columns={
                'text_completeness': 'news_quality_score',
                'has_source': 'news_source_count'
            })

            # 3. Calculate news freshness (hours since last news)
            news_timestamps = news_copy.index.to_series()
            
            # Merge with main DataFrame
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

            # Merge using merge_asof
            df_reset = df_enriched.reset_index()
            df_reset = df_reset.rename(columns={'index': 'datetime'} if 'index' in df_reset.columns else {})
            
            # ✅ Нормалізуємо timezone + precision в df_reset
            if 'datetime' in df_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(df_reset['datetime']):
                    if hasattr(df_reset['datetime'].dtype, 'tz') and df_reset['datetime'].dt.tz is not None:
                        df_reset['datetime'] = df_reset['datetime'].dt.tz_localize(None)
                    # Convert to ns precision
                    if df_reset['datetime'].dtype != 'datetime64[ns]':
                        df_reset['datetime'] = df_reset['datetime'].astype('datetime64[ns]')
            
            aggregated_reset = aggregated.reset_index()
            aggregated_reset = aggregated_reset.rename(columns={time_col: 'datetime'})
            
            # ✅ Нормалізуємо timezone + precision в aggregated_reset
            if 'datetime' in aggregated_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(aggregated_reset['datetime']):
                    if hasattr(aggregated_reset['datetime'].dtype, 'tz') and aggregated_reset['datetime'].dt.tz is not None:
                        aggregated_reset['datetime'] = aggregated_reset['datetime'].dt.tz_localize(None)
                    # Convert to ns precision
                    if aggregated_reset['datetime'].dtype != 'datetime64[ns]':
                        aggregated_reset['datetime'] = aggregated_reset['datetime'].astype('datetime64[ns]')

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

            # Calculate news freshness (hours since last news)
            df_merged['news_freshness_hours'] = 0.0
            
            # ✅ FIX: Normalize timezone для news_timestamps
            if news_timestamps.dt.tz is not None:
                news_timestamps = news_timestamps.dt.tz_localize(None)
            
            for idx in df_merged.index:
                # ✅ FIX: Normalize timezone для idx якщо потрібно
                idx_normalized = idx
                if hasattr(idx, 'tz') and idx.tz is not None:
                    idx_normalized = idx.tz_localize(None)
                
                # Find most recent news before this timestamp
                recent_news = news_timestamps[news_timestamps <= idx_normalized]
                if not recent_news.empty:
                    time_diff = (idx_normalized - recent_news.max()).total_seconds() / 3600
                    df_merged.loc[idx, 'news_freshness_hours'] = time_diff
                else:
                    df_merged.loc[idx, 'news_freshness_hours'] = 999.0  # No news available

            logger.info(f"✅ Added news quality features. Avg quality: {df_merged['news_quality_score'].mean():.2f}, Avg freshness: {df_merged['news_freshness_hours'].mean():.1f}h")
            return df_merged

        except Exception as e:
            logger.error(f"Error during news quality enrichment: {e}", exc_info=True)
            return df
