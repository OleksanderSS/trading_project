from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher
from src.features.nlp.extractors.entity_extractor import EntityExtractor
from src.features.nlp.extractors.keyword_extractor import KeywordExtractor

logger = ProjectLogger.get_logger("KeywordEntityEnricher")

# Constants to avoid duplication
DATETIME64_NS = "datetime64[ns]"
TEXT_COLUMNS = ['title', 'text', 'description', 'content']
TIME_COLUMNS = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']

class KeywordEntityEnricher(BaseEnricher):
    """
    Enriches DataFrame with keyword and entity features from news.
    Extracts keywords and named entities, then aggregates them per timestamp.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with optional config from FeatureOrchestrator."""
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        self.config = config or {}

        # Initialize keyword extractor
        keyword_config = self.config.get('keywords', {})
        self.keyword_extractor = KeywordExtractor(keyword_config)

        # Initialize entity extractor
        entity_config = self.config.get('entities', {
            'spacy_model': 'en_core_web_sm',
            'disable_components': ['parser', 'lemmatizer', 'attribute_ruler']
        })
        self.entity_extractor: EntityExtractor | None = None
        try:
            self.entity_extractor = EntityExtractor(entity_config)
        except Exception as e:
            logger.warning(f"Failed to initialize EntityExtractor: {e}. Entity features will be skipped.")
            self.entity_extractor = None

        self._analysis_cache = {}  # Cache for aggregated results
        logger.info("KeywordEntityEnricher initialized")

    @property
    def name(self) -> str:
        return "keyword_entity"

    @property
    def priority(self) -> int:
        """Run after NLP (30), before sentiment (40)"""
        return 35

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds keyword and entity features to the DataFrame.

        Args:
            df: Input DataFrame with DatetimeIndex
            **kwargs: Should contain 'news' DataFrame

        Returns:
            DataFrame with added keyword_count, entity_count, ticker_mentions features
        """
        if not self._validate_input(df):
            return df

        news_df = kwargs.get('news')
        if not self._validate_news_data(news_df):
            return df

        text_col = self._find_text_column(news_df)
        if text_col is None:
            return df

        time_col = self._find_time_column(news_df)
        if time_col is None:
            return df

        try:
            # ✅ OPTIMIZATION: Cache analysis results based on news_df hash
            import hashlib
            news_hash = hashlib.sha256(pd.util.hash_pandas_object(news_df, index=True).values.tobytes()).hexdigest()

            if news_hash in self._analysis_cache:
                logger.info("🚀 Using cached keyword/entity analysis results")
                aggregated = self._analysis_cache[news_hash]
                time_col = self._find_time_column(news_df) # Still need time_col for merge
            else:
                logger.info(f"🔄 Performing fresh keyword/entity analysis for {len(news_df)} items")
                news_copy = self._prepare_news_data(news_df, time_col)
                news_copy = self._extract_features(news_copy, text_col)
                aggregated = self._aggregate_by_time(news_copy, time_col)
                self._analysis_cache[news_hash] = aggregated

            return self._merge_with_main_df(df, aggregated, time_col)
        except Exception as e:
            logger.error(f"Error during keyword/entity enrichment: {e}", exc_info=True)
            return df

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame."""
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping keyword/entity enrichment.")
            return False
        return True

    def _validate_news_data(self, news_df: pd.DataFrame) -> bool:
        """Validate news data."""
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            logger.warning("No news data available in kwargs. Skipping keyword/entity enrichment.")
            return False
        return True

    def _find_text_column(self, news_df: pd.DataFrame) -> str | None:
        """Find text column in news DataFrame."""
        for col in TEXT_COLUMNS:
            if col in news_df.columns:
                return col

        logger.error("No text column found in news data. Skipping keyword/entity enrichment.")
        return None

    def _find_time_column(self, news_df: pd.DataFrame) -> str | None:
        """Find time column in news DataFrame."""
        for col in TIME_COLUMNS:
            if col in news_df.columns:
                return col

        logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping keyword/entity enrichment.")
        return None

    def _process_enrichment(self, df: pd.DataFrame, news_df: pd.DataFrame, text_col: str, time_col: str) -> pd.DataFrame:
        """Process the enrichment workflow."""
        news_copy = self._prepare_news_data(news_df, time_col)

        logger.info(f"✅ Found time column '{time_col}' with {len(news_copy)} valid timestamps")
        logger.info(f"Extracting keywords and entities from {len(news_copy)} news items...")

        news_copy = self._extract_features(news_copy, text_col)
        aggregated = self._aggregate_by_time(news_copy, time_col)

        return self._merge_with_main_df(df, aggregated, time_col)

    def _prepare_news_data(self, news_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Prepare news data with normalized datetime."""
        news_copy = news_df.copy()

        # Normalize timezone and convert to datetime64[ns]
        news_copy[time_col] = pd.to_datetime(news_copy[time_col], errors='coerce', utc=True)
        if news_copy[time_col].dt.tz is not None:
            news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
        news_copy[time_col] = news_copy[time_col].astype(DATETIME64_NS)

        return news_copy.dropna(subset=[time_col])

    def _extract_features(self, news_copy: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Extract keywords and entities from news."""
        # Extract keywords
        news_copy['keywords'] = news_copy[text_col].fillna('').apply(lambda x: self.keyword_extractor.extract(x))
        news_copy['keyword_count'] = news_copy['keywords'].apply(len)

        # Extract entities
        if self.entity_extractor:
            news_copy['entities'] = news_copy[text_col].fillna('').apply(
                lambda x: self.entity_extractor.extract(x, entity_types=['ORG', 'GPE', 'PERSON'])
            )
            news_copy['entity_count'] = news_copy['entities'].apply(len)
        else:
            news_copy['entity_count'] = 0

        return news_copy

    def _aggregate_by_time(self, news_copy: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Aggregate news data by time (hourly)."""
        news_copy = news_copy.set_index(time_col)

        # Resample to hourly and aggregate
        return news_copy.resample('1h').agg({
            'keyword_count': 'sum',
            'entity_count': 'sum'
        })

    def _merge_with_main_df(self, df: pd.DataFrame, aggregated: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Merge aggregated features with main DataFrame."""
        df_enriched = df.copy()

        # Ensure df has DatetimeIndex
        if not self._ensure_datetime_index(df_enriched):
            return df

        # Normalize timezones
        self._normalize_timezones(df_enriched, aggregated)

        # Prepare DataFrames for merge
        df_reset = self._prepare_df_for_merge(df_enriched)
        aggregated_reset = self._prepare_aggregated_for_merge(aggregated, time_col)

        # Merge using merge_asof for time-series alignment
        df_merged = pd.merge_asof(
            df_reset.sort_values('datetime'),
            aggregated_reset.sort_values('datetime'),
            on='datetime',
            direction='backward'
        )

        return self._finalize_merge_result(df_merged)

    def _ensure_datetime_index(self, df_enriched: pd.DataFrame) -> bool:
        """Ensure DataFrame has DatetimeIndex."""
        if isinstance(df_enriched.index, pd.DatetimeIndex):
            return True

        if 'datetime' in df_enriched.columns:
            df_enriched = df_enriched.set_index('datetime')
            return True

        logger.error("Cannot merge: df has no DatetimeIndex or 'datetime' column")
        return False

    def _normalize_timezones(self, df_enriched: pd.DataFrame, aggregated: pd.DataFrame):
        """Normalize timezones in both DataFrames."""
        if df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_localize(None)
        if aggregated.index.tz is not None:
            aggregated.index = aggregated.index.tz_localize(None)

    def _prepare_df_for_merge(self, df_enriched: pd.DataFrame) -> pd.DataFrame:
        """Prepare main DataFrame for merge."""
        df_reset = df_enriched.reset_index()
        df_reset = df_reset.rename(columns={'index': 'datetime'} if 'index' in df_reset.columns else {})

        # Normalize timezone + precision
        self._normalize_datetime_column(df_reset, 'datetime')

        return df_reset

    def _prepare_aggregated_for_merge(self, aggregated: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Prepare aggregated DataFrame for merge."""
        aggregated_reset = aggregated.reset_index()
        aggregated_reset = aggregated_reset.rename(columns={time_col: 'datetime'})

        # Normalize timezone + precision
        self._normalize_datetime_column(aggregated_reset, 'datetime')

        return aggregated_reset

    def _normalize_datetime_column(self, df: pd.DataFrame, col: str):
        """Normalize datetime column timezone and precision."""
        if col not in df.columns:
            return

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if hasattr(df[col].dtype, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

            if df[col].dtype != DATETIME64_NS:
                df[col] = df[col].astype(DATETIME64_NS)

    def _finalize_merge_result(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Finalize merge result."""
        df_merged = df_merged.set_index('datetime')
        df_merged['keyword_count'] = df_merged['keyword_count'].fillna(0).astype(int)
        df_merged['entity_count'] = df_merged['entity_count'].fillna(0).astype(int)

        logger.info(f"✅ Added keyword/entity features. Avg keywords: {df_merged['keyword_count'].mean():.1f}, Avg entities: {df_merged['entity_count'].mean():.1f}")
        return df_merged
