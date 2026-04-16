import pandas as pd
import logging
from typing import Dict, Any, Optional, List

from src.features.enrichers.base import BaseEnricher
from src.features.nlp.extractors.keyword_extractor import KeywordExtractor
from src.features.nlp.extractors.entity_extractor import EntityExtractor
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("KeywordEntityEnricher")

class KeywordEntityEnricher(BaseEnricher):
    """
    Enriches DataFrame with keyword and entity features from news.
    Extracts keywords and named entities, then aggregates them per timestamp.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config from FeatureOrchestrator."""
        self.config = config or {}
        
        # Initialize keyword extractor
        keyword_config = self.config.get('keywords', {})
        self.keyword_extractor = KeywordExtractor(keyword_config)
        
        # Initialize entity extractor
        entity_config = self.config.get('entities', {
            'spacy_model': 'en_core_web_sm',
            'disable_components': ['parser', 'lemmatizer', 'attribute_ruler']
        })
        try:
            self.entity_extractor = EntityExtractor(entity_config)
        except Exception as e:
            logger.warning(f"Failed to initialize EntityExtractor: {e}. Entity features will be skipped.")
            self.entity_extractor = None
        
        logger.info("KeywordEntityEnricher initialized")

    @property
    def name(self) -> str:
        return "keyword_entity"

    @property
    def priority(self) -> int:
        """Run after NLP (30), before sentiment (40)"""
        return 35

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds keyword and entity features to the DataFrame.

        Args:
            df: Input DataFrame with DatetimeIndex
            **kwargs: Should contain 'news' DataFrame

        Returns:
            DataFrame with added keyword_count, entity_count, ticker_mentions features
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping keyword/entity enrichment.")
            return df

        # Get news data from kwargs
        news_df = kwargs.get('news')
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            logger.warning("No news data available in kwargs. Skipping keyword/entity enrichment.")
            return df

        # Find text column
        text_col = None
        for col in ['title', 'text', 'description', 'content']:
            if col in news_df.columns:
                text_col = col
                break

        if text_col is None:
            logger.error("No text column found in news data. Skipping keyword/entity enrichment.")
            return df

        # Find time column
        time_col = None
        possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
        for col in possible_time_cols:
            if col in news_df.columns:
                time_col = col
                break

        if time_col is None:
            logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping keyword/entity enrichment.")
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

            logger.info(f"Extracting keywords and entities from {len(news_copy)} news items...")

            # Extract keywords and entities
            news_copy['keywords'] = news_copy[text_col].fillna('').apply(lambda x: self.keyword_extractor.extract(x))
            news_copy['keyword_count'] = news_copy['keywords'].apply(len)
            
            if self.entity_extractor:
                news_copy['entities'] = news_copy[text_col].fillna('').apply(
                    lambda x: self.entity_extractor.extract(x, entity_types=['ORG', 'GPE', 'PERSON'])
                )
                news_copy['entity_count'] = news_copy['entities'].apply(len)
            else:
                news_copy['entity_count'] = 0

            # Aggregate by time (hourly)
            news_copy = news_copy.set_index(time_col)
            
            # Resample to hourly and aggregate
            aggregated = news_copy.resample('1h').agg({
                'keyword_count': 'sum',
                'entity_count': 'sum'
            })

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

            # Merge using merge_asof for time-series alignment
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
            df_merged['keyword_count'] = df_merged['keyword_count'].fillna(0).astype(int)
            df_merged['entity_count'] = df_merged['entity_count'].fillna(0).astype(int)

            logger.info(f"✅ Added keyword/entity features. Avg keywords: {df_merged['keyword_count'].mean():.1f}, Avg entities: {df_merged['entity_count'].mean():.1f}")
            return df_merged

        except Exception as e:
            logger.error(f"Error during keyword/entity enrichment: {e}", exc_info=True)
            return df
