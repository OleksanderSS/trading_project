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
        news_df = kwargs.get('news_data')

        if news_df is None or news_df.empty:
            logger.warning("No news data provided for NLP enrichment. Skipping.")
            return df

        if 'ticker' not in df.columns:
            logger.error("Main DataFrame missing 'ticker' column. NLP enrichment aborted.")
            return df

        logger.info(f"Starting NLP analysis for {len(news_df)} news items...")

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
            
            features_to_merge = analyzed_news[available_cols + ['ticker']].copy()
            
            # Reset index to make 'datetime' a column for merging if it's the index
            if isinstance(features_to_merge.index, pd.DatetimeIndex):
                features_to_merge = features_to_merge.reset_index().rename(columns={features_to_merge.index.name: 'datetime'})
            
            # Apply prefix
            rename_map = {col: f"nlp_{col}" for col in available_cols}
            features_to_merge = features_to_merge.rename(columns=rename_map)

            # 3. Merge with main DataFrame
            # We use merge_asof if data is time-series, or a standard left merge if exact alignment is expected
            df_enriched = df.copy()
            if not isinstance(df_enriched.index, pd.DatetimeIndex):
                df_enriched.index = pd.to_datetime(df_enriched.index)
            
            df_enriched = df_enriched.sort_index()
            features_to_merge = features_to_merge.sort_values('datetime')

            # Grouped merge to ensure ticker-specific alignment
            result_dfs = []
            for ticker, group in df_enriched.groupby('ticker'):
                ticker_features = features_to_merge[features_to_merge['ticker'] == ticker]
                
                if ticker_features.empty:
                    result_dfs.append(group)
                    continue
                
                # Align news features to the closest preceding price timestamp
                merged_group = pd.merge_asof(
                    group, 
                    ticker_features.drop(columns=['ticker']), 
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