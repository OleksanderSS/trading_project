# src/features/enrichers/nlp_features_enricher.py

import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.features.enrichers.base import BaseEnricher
from src.features.nlp.processors.news_analyzer import NewsAnalyzer
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config

logger = ProjectLogger.get_logger("NLPFeaturesEnricher")

class NLPFeaturesEnricher(BaseEnricher):
    """
    Enriches the main DataFrame with NLP-based features derived from news.
    Uses NewsAnalyzer to process raw news text into sentiment and cluster scores.
    """

    def __init__(self):
        self.config = get_current_config().get('enrichment.nlp_features', {})
        self.analyzer = NewsAnalyzer(
            n_clusters=self.config.get('n_clusters', 5),
            max_features=self.config.get('max_features', 1000)
        )
        logger.info(f"NLPFeaturesEnricher initialized with {self.config.get('n_clusters', 5)} clusters.")

    @property
    def name(self) -> str:
        return "nlp_features"

    @property
    def priority(self) -> int:
        """
        Determines the execution order in the FeatureOrchestrator.
        Set to 30 to run after TechnicalAnalysis (20) but before SentimentFeatures (40).
        """
        return 30

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Processes news data and merges NLP features into the main price DataFrame.

        Args:
            df: The main price DataFrame (must have DatetimeIndex and 'ticker' column).
            **kwargs: Should contain 'news_data' (pd.DataFrame).

        Returns:
            DataFrame with 'nlp_' prefixed sentiment and clustering features.
        """
        news_df = kwargs.get('news')  # ✅ Виправлено з 'news_data' на 'news'

        if news_df is None or news_df.empty:
            logger.warning("No news data provided for NLP enrichment. Skipping.")
            return df

        if 'ticker' not in df.columns:
            logger.error("Main DataFrame missing 'ticker' column. NLP enrichment aborted.")
            return df

        logger.info(f"Starting NLP analysis for {len(news_df)} news items...")
        logger.info(f"News columns: {news_df.columns.tolist()}")

        try:
            # 1. Perform news analysis (clustering and sentiment)
            # NewsAnalyzer expects specific columns and handles date conversion
            analyzed_news = self.analyzer.cluster_news(
                news_df, 
                text_column=self.config.get('text_column', 'title'),
                date_column=self.config.get('date_column', 'published_at')
            )

            if analyzed_news.empty:
                logger.warning("NewsAnalyzer returned empty results.")
                return df

            # 2. Prepare features for merging
            # We select relevant columns and add the 'nlp_' prefix
            nlp_cols = ['sentiment_score', 'subjectivity_score', 'cluster']
            available_cols = [c for c in nlp_cols if c in analyzed_news.columns]
            
            # Include ticker if available, but DON'T filter by columns yet (keep index)
            features_to_merge = analyzed_news.copy()
            
            # Reset index to make 'datetime' a column for merging if it's the index
            if isinstance(features_to_merge.index, pd.DatetimeIndex):
                # Створюємо нову колонку 'datetime' з індексу
                features_to_merge['datetime'] = features_to_merge.index
                features_to_merge = features_to_merge.reset_index(drop=True)
                # ✅ Нормалізуємо timezone: конвертуємо в UTC і видаляємо timezone info
                features_to_merge['datetime'] = pd.to_datetime(features_to_merge['datetime']).dt.tz_localize(None)
                # ✅ Конвертуємо в ns для сумісності з pandas merge
                features_to_merge['datetime'] = features_to_merge['datetime'].astype('datetime64[ns]')
                logger.info(f"Created 'datetime' column from DatetimeIndex (tz-naive, ns precision)")
            elif 'datetime' not in features_to_merge.columns:
                # ✅ FIX: Шукаємо будь-яку колонку з датою та конвертуємо в datetime
                date_col = None
                possible_date_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp']
                for col in possible_date_cols:
                    if col in features_to_merge.columns:
                        date_col = col
                        break
                
                if date_col:
                    features_to_merge['datetime'] = pd.to_datetime(features_to_merge[date_col])
                    # ✅ Нормалізуємо timezone
                    if hasattr(features_to_merge['datetime'].dtype, 'tz') and features_to_merge['datetime'].dt.tz is not None:
                        features_to_merge['datetime'] = features_to_merge['datetime'].dt.tz_localize(None)
                    features_to_merge['datetime'] = features_to_merge['datetime'].astype('datetime64[ns]')
                    logger.info(f"✅ Created 'datetime' column from '{date_col}' (tz-naive, ns precision)")
                else:
                    logger.error(f"❌ No datetime column found. Columns: {features_to_merge.columns.tolist()}")
                    return df
            
            # Check if datetime column exists
            if 'datetime' not in features_to_merge.columns:
                logger.error(f"No 'datetime' column after processing. Columns: {features_to_merge.columns.tolist()}")
                return df
            
            # Now filter to only needed columns
            keep_cols = ['datetime'] + available_cols + (['ticker'] if 'ticker' in features_to_merge.columns else [])
            features_to_merge = features_to_merge[keep_cols]
            
            # Apply prefix
            rename_map = {col: f"nlp_{col}" for col in available_cols}
            features_to_merge = features_to_merge.rename(columns=rename_map)
            
            # ✅ Нормалізуємо timezone в features_to_merge
            if 'datetime' in features_to_merge.columns:
                if pd.api.types.is_datetime64_any_dtype(features_to_merge['datetime']):
                    if hasattr(features_to_merge['datetime'].dtype, 'tz') and features_to_merge['datetime'].dt.tz is not None:
                        features_to_merge['datetime'] = features_to_merge['datetime'].dt.tz_localize(None)
                        logger.info("Removed timezone from features_to_merge for merge compatibility")
                    # ✅ Convert to ns precision
                    if features_to_merge['datetime'].dtype != 'datetime64[ns]':
                        features_to_merge['datetime'] = features_to_merge['datetime'].astype('datetime64[ns]')
                        logger.info("Converted features_to_merge to ns precision")

            # 3. Merge with main DataFrame
            # We use merge_asof if data is time-series, or a standard left merge if exact alignment is expected
            df_enriched = df.copy()
            if not isinstance(df_enriched.index, pd.DatetimeIndex):
                if 'datetime' in df_enriched.columns:
                    df_enriched = df_enriched.set_index('datetime')
                else:
                    df_enriched.index = pd.to_datetime(df_enriched.index)
            
            # ✅ Нормалізуємо timezone в df_enriched
            if df_enriched.index.tz is not None:
                df_enriched.index = df_enriched.index.tz_localize(None)
                logger.info("Removed timezone from df index for merge compatibility")
            
            # ✅ Конвертуємо в ns precision
            if df_enriched.index.dtype != 'datetime64[ns]':
                df_enriched.index = df_enriched.index.astype('datetime64[ns]')
                logger.info("Converted df index to ns precision")
            
            df_enriched = df_enriched.sort_index()
            features_to_merge = features_to_merge.sort_values('datetime')

            # Grouped merge to ensure ticker-specific alignment
            result_dfs = []
            for ticker, group in df_enriched.groupby('ticker'):
                if 'ticker' in features_to_merge.columns:
                    ticker_features = features_to_merge[features_to_merge['ticker'] == ticker]
                else:
                    # Global news applies to all tickers
                    ticker_features = features_to_merge.copy()
                
                # Prevent merge_asof Duplicate Key ValueErrors
                ticker_features = ticker_features.drop_duplicates(subset=['datetime'], keep='last')
                
                if ticker_features.empty:
                    result_dfs.append(group)
                    continue
                
                # Align news features to the closest preceding price timestamp
                drop_cols = ['ticker'] if 'ticker' in ticker_features.columns else []
                merged_group = pd.merge_asof(
                    group, 
                    ticker_features.drop(columns=drop_cols) if drop_cols else ticker_features, 
                    left_index=True, 
                    right_on='datetime', 
                    direction='backward'
                )
                # Restore the original index
                merged_group.set_index('datetime', inplace=True)
                result_dfs.append(merged_group)

            final_df = pd.concat(result_dfs).sort_index()
            
            logger.info(f"NLP enrichment complete. Added features: {list(rename_map.values())}")
            return final_df

        except Exception as e:
            logger.error(f"Error during NLP feature enrichment: {e}", exc_info=True)
            return df