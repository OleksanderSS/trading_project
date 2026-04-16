import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.features.enrichers.base import BaseEnricher
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.core.logging.logger import ProjectLogger

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config from FeatureOrchestrator."""
        self.config = config or {}
        self.analyzer = NewsImpactAnalyzer(self.config)
        logger.info("NewsImpactEnricher initialized")

    @property
    def name(self) -> str:
        return "news_impact"

    @property
    def priority(self) -> int:
        """Run after NLP (30) and sentiment (40), before significance (70)"""
        return 45

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
            logger.warning("Input DataFrame is empty. Skipping news impact enrichment.")
            return df

        # Debug logging
        logger.debug(f"📊 NewsImpactEnricher.enrich() called. DataFrame shape: {df.shape}")
        logger.debug(f"📊 Columns in df: {df.columns.tolist()[:20]}")
        logger.debug(f"📊 Has news_title: {'news_title' in df.columns}")
        logger.debug(f"📊 Has news_sentiment: {'news_sentiment' in df.columns}")
        logger.debug(f"📊 kwargs keys: {kwargs.keys()}")

        # Check if this is event-centric format (news columns in df itself)
        is_event_centric = 'news_title' in df.columns or 'news_sentiment' in df.columns
        
        if is_event_centric:
            logger.info("✅ Detected event-centric format. Processing news from DataFrame columns...")
            return self._enrich_event_centric(df)
        
        # Traditional format: get news data from kwargs
        news_df = kwargs.get('news')
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            logger.warning("⚠️ No news data available in kwargs and not event-centric format. Skipping news impact enrichment.")
            return df
        
        logger.info("📰 Using traditional format with separate news DataFrame...")
        return self._enrich_traditional(df, news_df)

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
            
            # Find text column
            text_col = None
            for col in ['news_title', 'news_text', 'title', 'text', 'description']:
                if col in df_enriched.columns:
                    text_col = col
                    break
            
            logger.debug(f"📝 Text column found: {text_col}")
            if text_col is None:
                logger.warning("⚠️ No text column found in event-centric DataFrame. Skipping news impact enrichment.")
                return df
            
            # Find time column
            time_col = None
            for col in ['datetime', 'published_at', 'publishedAt', 'published_date', 'timestamp', 'date']:
                if col in df_enriched.columns:
                    time_col = col
                    break
            
            logger.debug(f"⏰ Time column found: {time_col}")
            if time_col is None:
                logger.warning("⚠️ No time column found in event-centric DataFrame. Skipping news impact enrichment.")
                return df
            
            # Prepare news data for analyzer
            # Filter out rows without news (where news_title is NaN)
            news_rows = df_enriched[df_enriched[text_col].notna()].copy()
            
            logger.info(f"📰 Found {len(news_rows)} news rows out of {len(df_enriched)} total rows")
            
            if news_rows.empty:
                logger.warning("⚠️ No news rows found in event-centric DataFrame. Adding zero impact scores.")
                df_enriched['news_impact_score'] = 0.0
                df_enriched['news_significance_level'] = 0
                return df_enriched
            
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
                logger.warning("⚠️ No valid news data after preparation. Adding zero impact scores.")
                df_enriched['news_impact_score'] = 0.0
                df_enriched['news_significance_level'] = 0
                return df_enriched
            
            logger.info(f"🚀 Running NewsImpactAnalyzer for {len(news_prepared)} news items...")
            
            # Run analyzer
            results = self.analyzer.analyze(news_prepared)
            
            logger.debug(f"📈 Analyzer results keys: {results.keys() if results else 'None'}")
            
            if not results:
                logger.warning("⚠️ NewsImpactAnalyzer returned no results. Adding zero impact scores.")
                df_enriched['news_impact_score'] = 0.0
                df_enriched['news_significance_level'] = 0
                return df_enriched
            
            impact_scores = results.get('news_impact_scores')
            significance_levels = results.get('news_significance_levels')
            
            logger.debug(f"📊 Impact scores type: {type(impact_scores)}, shape: {impact_scores.shape if hasattr(impact_scores, 'shape') else 'N/A'}")
            
            if impact_scores is None or impact_scores.empty:
                logger.warning("⚠️ No impact scores generated. Adding zero impact scores.")
                df_enriched['news_impact_score'] = 0.0
                df_enriched['news_significance_level'] = 0
                return df_enriched
            
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
            
            logger.info(f"✅ Added news impact features (event-centric). Impact score range: [{df_enriched['news_impact_score'].min():.3f}, {df_enriched['news_impact_score'].max():.3f}]")
            logger.info(f"✅ Significance level range: [{df_enriched['news_significance_level'].min()}, {df_enriched['news_significance_level'].max()}]")
            return df_enriched
            
        except Exception as e:
            logger.error(f"❌ Error during event-centric news impact enrichment: {e}", exc_info=True)
            # Add zero scores on error
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
        # Find text column
        text_col = None
        for col in ['text', 'title', 'description', 'content']:
            if col in news_df.columns:
                text_col = col
                break

        if text_col is None:
            logger.error("No text column found in news data. Skipping news impact enrichment.")
            return df

        # Find time column
        time_col = None
        possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
        for col in possible_time_cols:
            if col in news_df.columns:
                time_col = col
                break

        if time_col is None:
            logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping news impact enrichment.")
            return df

        try:
            # Prepare news data for analyzer
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

            # Run analyzer
            results = self.analyzer.analyze(news_prepared)

            if not results:
                logger.warning("NewsImpactAnalyzer returned no results.")
                return df

            impact_scores = results.get('news_impact_scores')
            significance_levels = results.get('news_significance_levels')

            if impact_scores is None or impact_scores.empty:
                logger.warning("No impact scores generated.")
                return df

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
            if impact_scores.index.tz is not None:
                impact_scores.index = impact_scores.index.tz_localize(None)
            if significance_levels is not None and hasattr(significance_levels, 'index') and significance_levels.index.tz is not None:
                significance_levels.index = significance_levels.index.tz_localize(None)

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

        except Exception as e:
            logger.error(f"Error during news impact enrichment: {e}", exc_info=True)
            return df
