from typing import Any

import pandas as pd

from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("NewsImpactEnricher")

class NewsImpactEnricher(BaseEnricher):
    """
    Enriches DataFrame with news impact scores using NewsImpactAnalyzer.
    This is the "reverse impact analyzer" that calculates sentiment-based,
    time-decaying impact scores from news.

    Supports two modes:
    1. Event-centric format: news columns (news_title, news_sentiment) are in df itself
    2. Traditional format: separate 'news' DataFrame in kwargs
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with optional config from FeatureOrchestrator."""
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        self.config = config or {}
        self.analyzer = NewsImpactAnalyzer(self.config)
        self._analysis_cache: dict[str, Any] = {}  # Cache for results
        self.last_known_scores: dict[str, Any] = {}
        logger.info("NewsImpactEnricher initialized")

    @property
    def name(self) -> str:
        return "news_impact"

    @property
    def priority(self) -> int:
        """Run after NLP (30) and sentiment (40), before significance (70)"""
        return 45

    def _has_news_columns(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains news columns for event-centric processing."""
        return 'news_title' in df.columns or 'news_sentiment' in df.columns

    def _update_last_known_scores(self, df: pd.DataFrame) -> None:
        """Update last known scores for future use."""
        self.last_known_scores = {
            'news_impact_score': df['news_impact_score'].iloc[-1] if 'news_impact_score' in df.columns else 0.0,
            'news_significance_level': df['news_significance_level'].iloc[-1] if 'news_significance_level' in df.columns else 'low'
        }

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds news impact features to the DataFrame.

        Supports two modes:
        1. Event-centric format: news columns (news_title, news_sentiment) are in df itself
        2. Traditional format: separate 'news' DataFrame in kwargs

        Args:
            df: Input DataFrame with DatetimeIndex or datetime column
            **kwargs: May contain 'news' DataFrame with 'text' or 'title' column

        Returns:
            DataFrame with added news_impact_score and news_significance_level columns
        """
        if df.empty:
            return df

        try:
            # First try traditional format with news data from kwargs
            news_df = kwargs.get('news')
            if news_df is not None and isinstance(news_df, pd.DataFrame) and not news_df.empty:
                logger.info(f"📰 Using traditional format with news DataFrame ({len(news_df)} rows)...")
                return self._enrich_traditional(df, news_df)

            # Check if we have sentiment features from SentimentFeaturesEnricher
            sentiment_cols = ['nlp_sentiment_score', 'sentiment_sma_5', 'sentiment_velocity', 'news_intensity']
            has_sentiment = any(col in df.columns for col in sentiment_cols)

            if has_sentiment:
                logger.info("📰 Using sentiment features from SentimentFeaturesEnricher...")
                return self._enrich_from_sentiment_features(df)

            # Check if we have event-centric format (news columns in df)
            if self._has_news_columns(df):
                logger.debug("📰 Using event-centric format (news columns in DataFrame)...")
                return self._enrich_event_centric(df)

            logger.warning("⚠️ No news data available and no sentiment features found. Skipping news impact enrichment.")
            return df

        except Exception as e:
            logger.error(f"❌ Error during news impact enrichment: {e}", exc_info=True)
            return self._add_zero_scores(df)

    def _enrich_from_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich DataFrame using sentiment features created by SentimentFeaturesEnricher.

        Args:
            df: DataFrame with sentiment features (nlp_sentiment_score, sentiment_sma_5, etc.)

        Returns:
            DataFrame with added news_impact_score and news_significance_level columns
        """
        try:
            df_enriched = df.copy()

            # Find the best sentiment column
            sentiment_col = None
            for col in ['nlp_sentiment_score', 'sentiment_sma_5', 'sentiment_velocity', 'news_intensity']:
                if col in df_enriched.columns:
                    non_zero_count = (df_enriched[col] != 0).sum()
                    if non_zero_count > 0:
                        sentiment_col = col
                        logger.info(f"📝 Using sentiment column '{col}' with {non_zero_count} non-zero values")
                        break

            if sentiment_col is None:
                logger.warning("⚠️ No non-zero sentiment features found. Adding zero scores.")
                return self._add_zero_scores(df_enriched)

            # Calculate impact scores based on sentiment
            df_enriched['news_impact_score'] = df_enriched[sentiment_col].abs()
            df_enriched['news_significance_level'] = pd.cut(
                df_enriched['news_impact_score'],
                bins=[-1, 0.1, 0.3, 0.6, 1.0],
                labels=[0, 1, 2, 3]
            ).fillna(0).astype(int)

            # Log results
            non_zero_impact = (df_enriched['news_impact_score'] > 0).sum()
            max_impact = df_enriched['news_impact_score'].max()
            logger.info(f"✅ Created news impact scores: {non_zero_impact}/{len(df_enriched)} non-zero, max={max_impact:.3f}")

            return df_enriched

        except Exception as e:
            logger.error(f"❌ Error during sentiment features enrichment: {e}", exc_info=True)
            return self._add_zero_scores(df)

    def _enrich_event_centric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches event-centric DataFrame where news columns are already present.

        Args:
            df: DataFrame with news_title, news_sentiment, datetime/published_at columns

        Returns:
            DataFrame with added news_impact_score and news_significance_level columns
        """
        try:
            df_enriched = df.copy()

            # Find required columns
            text_col = self._find_text_column(df_enriched)
            if text_col is None:
                return df

            time_col = self._find_time_column(df_enriched)
            if time_col is None:
                return df

            # Extract and prepare news data
            news_rows = self._extract_news_rows(df_enriched, text_col)
            if news_rows is None:
                return self._add_zero_scores(df_enriched)

            # Prepare news for analyzer
            news_prepared = self._prepare_news_for_analyzer(news_rows, text_col, time_col)
            if news_prepared is None:
                return self._add_zero_scores(df_enriched)

            # Run analyzer and merge results
            return self._run_analyzer_and_merge(df_enriched, news_prepared, time_col, "event-centric")

        except Exception as e:
            logger.error(f"❌ Error during event-centric news impact enrichment: {e}", exc_info=True)
            return self._add_zero_scores(df)

    def _find_text_column(self, df: pd.DataFrame) -> str | None:
        """Find the text column in DataFrame."""
        # First try traditional news columns
        for col in ['news_title', 'news_text', 'title', 'text', 'description']:
            if col in df.columns:
                logger.debug(f"📝 Text column found: {col}")
                return col

        # Then try existing sentiment/news columns (for our current data format)
        for col in ['nlp_sentiment_score', 'news_intensity', 'consumer_sentiment']:
            if col in df.columns:
                logger.debug(f"📝 Sentiment column found: {col}")
                return col

        # Try to get sentiment from news kwargs if available
        logger.warning("⚠️ No text or sentiment column found in DataFrame. Checking for news data in kwargs.")
        return None

    def _find_time_column(self, df: pd.DataFrame) -> str | None:
        """Find the time column in DataFrame."""
        for col in ['datetime', 'published_at', 'publishedAt', 'published_date', 'timestamp', 'date']:
            if col in df.columns:
                logger.debug(f"⏰ Time column found: {col}")
                return col

        logger.warning("⚠️ No time column found in DataFrame. Skipping news impact enrichment.")
        return None

    def _extract_news_rows(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame | None:
        """Extract rows with news data."""
        # For sentiment columns, we need to find rows with non-zero sentiment
        if 'sentiment' in text_col.lower() or 'nlp_sentiment' in text_col.lower():
            # Find rows with non-zero sentiment (meaningful news impact)
            news_rows = df[df[text_col] != 0].copy() if df[text_col].dtype in ['float64', 'int64'] else df[df[text_col].notna()].copy()
        else:
            # Traditional text columns
            news_rows = df[df[text_col].notna()].copy()

        logger.info(f"📰 Found {len(news_rows)} news rows out of {len(df)} total rows")

        if news_rows.empty:
            logger.warning("⚠️ No news rows found in DataFrame.")
            return None

        return news_rows

    def _prepare_news_for_analyzer(self, news_rows: pd.DataFrame, text_col: str, time_col: str) -> pd.DataFrame | None:
        """Prepare news data for the analyzer."""
        logger.info(f"🔍 Analyzing news impact for {len(news_rows)} news items...")

        # Prepare for analyzer
        news_prepared = news_rows[[text_col, time_col]].copy()
        news_prepared['text'] = news_prepared[text_col].fillna('')
        news_prepared[time_col] = pd.to_datetime(news_prepared[time_col], errors='coerce')
        news_prepared = news_prepared.dropna(subset=[time_col])
        news_prepared = news_prepared.set_index(time_col)
        news_prepared = news_prepared.sort_index()

        logger.debug(f"📊 News prepared shape: {news_prepared.shape}")

        if news_prepared.empty:
            logger.warning("⚠️ No valid news data after preparation.")
            return None

        return news_prepared

    def _run_analyzer_and_merge(self, df: pd.DataFrame, news_prepared: pd.DataFrame, time_col: str, mode: str) -> pd.DataFrame:
        """Run analyzer and merge results with DataFrame."""
        logger.info(f"🚀 Running NewsImpactAnalyzer for {len(news_prepared)} news items...")

        # Run analyzer
        results = self.analyzer.analyze(news_prepared)

        if not results:
            logger.warning("⚠️ NewsImpactAnalyzer returned no results.")
            return self._add_zero_scores(df)

        impact_scores = results.get('news_impact_scores')
        significance_levels = results.get('news_significance_levels')

        if impact_scores is None or impact_scores.empty:
            logger.warning("⚠️ No impact scores generated.")
            return self._add_zero_scores(df)

        # Merge results with DataFrame
        df_enriched = self._merge_impact_scores(df, impact_scores, significance_levels, time_col, mode)

        # Update last known scores for future use
        self._update_last_known_scores(df_enriched)

        return df_enriched

    def _merge_impact_scores(self, df: pd.DataFrame, impact_scores: pd.Series, significance_levels: pd.Series, time_col: str, mode: str) -> pd.DataFrame:
        """Merge impact scores with DataFrame."""
        df_enriched = df.copy()

        # Prepare datetime index for merging
        df_enriched[time_col] = pd.to_datetime(df_enriched[time_col], errors='coerce')

        # Normalize timezones
        if hasattr(df_enriched[time_col].dtype, 'tz') and df_enriched[time_col].dt.tz is not None:
            df_enriched[time_col] = df_enriched[time_col].dt.tz_localize(None)
        if impact_scores.index.tz is not None:
            impact_scores.index = impact_scores.index.tz_localize(None)
        if significance_levels is not None and hasattr(significance_levels, 'index') and significance_levels.index.tz is not None:
            significance_levels.index = significance_levels.index.tz_localize(None)

        # Create temporary index for merging
        df_enriched['_temp_idx'] = df_enriched[time_col]
        df_enriched = df_enriched.set_index('_temp_idx')

        # Merge impact scores (forward fill for time-decaying effect)
        impact_scores_aligned = impact_scores.reindex(df_enriched.index, method='ffill')
        df_enriched['news_impact_score'] = impact_scores_aligned.fillna(0.0)

        if significance_levels is not None:
            significance_aligned = significance_levels.reindex(df_enriched.index, method='ffill')
            # Convert categorical to numeric for ML models
            significance_map = {'low': 0, 'medium': 1, 'high': 2}
            df_enriched['news_significance_level'] = significance_aligned.map(significance_map).fillna(0).astype(int)
        else:
            df_enriched['news_significance_level'] = 0

        # Reset index
        df_enriched = df_enriched.reset_index(drop=True)

        logger.info(f"✅ Added news impact features ({mode}). Impact score range: [{df_enriched['news_impact_score'].min():.3f}, {df_enriched['news_impact_score'].max():.3f}]")
        logger.info(f"✅ Significance level range: [{df_enriched['news_significance_level'].min()}, {df_enriched['news_significance_level'].max()}]")

        return df_enriched

    def _add_zero_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add zero impact scores to DataFrame."""
        df_enriched = df.copy()
        df_enriched['news_impact_score'] = 0.0
        df_enriched['news_significance_level'] = 0
        return df_enriched

    def _enrich_traditional(self, df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches traditional format where news is a separate DataFrame.

        Args:
            df: Main DataFrame with DatetimeIndex or datetime column
            news_df: Separate news DataFrame with text and time columns

        Returns:
            DataFrame with added news_impact_score and news_significance_level columns
        """
        # Find required columns in news data
        text_col = self._find_news_text_column(news_df)
        if text_col is None:
            return df

        time_col = self._find_news_time_column(news_df)
        if time_col is None:
            return df

        try:
            # ✅ OPTIMIZATION: Cache analysis results based on news_df hash
            import hashlib
            news_hash = hashlib.sha256(pd.util.hash_pandas_object(news_df, index=True).values.tobytes()).hexdigest()

            if news_hash in self._analysis_cache:
                logger.info("🚀 Using cached news impact analysis results")
                results = self._analysis_cache[news_hash]
            else:
                logger.info(f"🔄 Performing fresh news impact analysis for {len(news_df)} items")
                # Prepare news data for analyzer
                news_prepared = self._prepare_traditional_news(news_df, text_col, time_col)
                if news_prepared is None:
                    return df
                # Run analyzer
                results = self.analyzer.analyze(news_prepared)
                self._analysis_cache[news_hash] = results

            # Run and merge results
            return self._run_analyzer_and_merge_traditional_results(df, results)

        except Exception as e:
            logger.error(f"Error during news impact enrichment: {e}", exc_info=True)
            return df

    def _find_news_text_column(self, news_df: pd.DataFrame) -> str | None:
        """Find the text column in news DataFrame."""
        for col in ['text', 'title', 'description', 'content']:
            if col in news_df.columns:
                return col

        logger.error("No text column found in news data. Skipping news impact enrichment.")
        return None

    def _find_news_time_column(self, news_df: pd.DataFrame) -> str | None:
        """Find the time column in news DataFrame."""
        possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
        for col in possible_time_cols:
            if col in news_df.columns:
                return col

        logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping news impact enrichment.")
        return None

    def _prepare_traditional_news(self, news_df: pd.DataFrame, text_col: str, time_col: str) -> pd.DataFrame | None:
        """Prepare traditional news data for analyzer."""
        try:
            news_prepared = news_df.copy()
            news_prepared['text'] = news_prepared[text_col].fillna('')
            # ✅ FIX: Normalize timezone and convert to datetime64[ns]
            news_prepared[time_col] = pd.to_datetime(news_prepared[time_col], errors='coerce', utc=True)
            if news_prepared[time_col].dt.tz is not None:
                news_prepared[time_col] = news_prepared[time_col].dt.tz_localize(None)
            news_prepared[time_col] = news_prepared[time_col].astype('datetime64[ns]')
            news_prepared = news_prepared.dropna(subset=[time_col])

            logger.info(f"✅ Found time column '{time_col}' with {len(news_prepared)} valid timestamps")
            news_prepared = news_prepared.set_index(time_col)
            news_prepared = news_prepared.sort_index()

            logger.info(f"Analyzing news impact for {len(news_prepared)} news items...")
            return news_prepared

        except Exception as e:
            logger.error(f"Error preparing traditional news data: {e}", exc_info=True)
            return None

    def _run_analyzer_and_merge_traditional_results(self, df: pd.DataFrame, results: dict[str, Any]) -> pd.DataFrame:
        """Run analyzer and merge results for traditional format."""
        if not results:
            logger.warning("NewsImpactAnalyzer returned no results.")
            return df

        impact_scores = results.get('news_impact_scores')
        significance_levels = results.get('news_significance_levels')

        if impact_scores is None or impact_scores.empty:
            logger.warning("No impact scores generated.")
            return df

        # Prepare main DataFrame for merging
        df_enriched = self._prepare_main_dataframe(df)
        if df_enriched is None:
            return df

        # Normalize timezones
        df_enriched, impact_scores, significance_levels = self._normalize_timezones_traditional(
            df_enriched, impact_scores, significance_levels
        )

        # Merge impact scores
        return self._merge_traditional_impact_scores(df_enriched, impact_scores, significance_levels)

    def _prepare_main_dataframe(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Prepare main DataFrame for merging."""
        df_enriched = df.copy()

        # Ensure df has DatetimeIndex
        if not isinstance(df_enriched.index, pd.DatetimeIndex):
            if 'datetime' in df_enriched.columns:
                df_enriched = df_enriched.set_index('datetime')
            else:
                logger.error("Cannot merge: df has no DatetimeIndex or 'datetime' column")
                return None

        return df_enriched

    def _normalize_timezones_traditional(self, df_enriched: pd.DataFrame, impact_scores: pd.Series, significance_levels: pd.Series) -> tuple:
        """Normalize timezones for traditional format."""
        # Normalize timezones
        if df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_localize(None)
        if impact_scores.index.tz is not None:
            impact_scores.index = impact_scores.index.tz_localize(None)
        if significance_levels is not None and hasattr(significance_levels, 'index') and significance_levels.index.tz is not None:
            significance_levels.index = significance_levels.index.tz_localize(None)

        return df_enriched, impact_scores, significance_levels

    def _merge_traditional_impact_scores(self, df_enriched: pd.DataFrame, impact_scores: pd.Series, significance_levels: pd.Series) -> pd.DataFrame:
        """Merge impact scores for traditional format."""
        # Reindex to match df's index (forward fill for time-decaying effect)
        impact_scores_aligned = impact_scores.reindex(df_enriched.index, method='ffill')

        # Add features
        df_enriched['news_impact_score'] = impact_scores_aligned.fillna(0.0)

        if significance_levels is not None:
            significance_aligned = significance_levels.reindex(df_enriched.index, method='ffill')
            # Convert categorical to numeric for ML models
            significance_map = {'low': 0, 'medium': 1, 'high': 2}
            df_enriched['news_significance_level'] = significance_aligned.map(significance_map).fillna(0).astype(int)

        logger.info(f"✅ Added news impact features (traditional). Impact score range: [{df_enriched['news_impact_score'].min():.3f}, {df_enriched['news_impact_score'].max():.3f}]")
        return df_enriched
