import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from src.config.unified_config_manager import get_current_config
from src.features.enrichers.base import BaseEnricher
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SentimentFeaturesEnricher")

class SentimentFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with advanced sentiment features derived from news scores.
    Calculates rolling statistics, momentum, intensity, and decay-weighted sentiment.
    """

    @property
    def name(self) -> str:
        return "sentiment_features"

    @property
    def priority(self) -> int:
        """
        Determines the execution order in the FeatureOrchestrator.
        Set to 40 to run after NLPFeaturesEnricher (30).
        """
        return 40

    def __init__(self):
        """
        Initializes the enricher by loading settings from the unified configuration.
        """
        config_manager = get_current_config()
        self.sentiment_config = config_manager.get('enrichment.sentiment', {})
        
        # Default windows if not provided in config
        self.windows = self.sentiment_config.get('windows', [5, 20, 50])
        self.decay_factor = self.sentiment_config.get('decay_factor', 0.95)
        self.enabled_features = self.sentiment_config.get('enabled_features', [
            'rolling_mean', 'rolling_std', 'velocity', 'intensity', 'decay_weighted'
        ])
        
        logger.info(f"SentimentFeaturesEnricher initialized with windows: {self.windows}")

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds sentiment-based features to the input DataFrame.

        Args:
            df: DataFrame containing at least 'datetime', 'ticker', and 'nlp_sentiment_score'.
            **kwargs: Additional parameters.

        Returns:
            DataFrame with additional sentiment features.
        """
        if df.empty:
            logger.warning("Received an empty DataFrame for sentiment enrichment.")
            return df

        if 'nlp_sentiment_score' not in df.columns or 'ticker' not in df.columns:
            logger.error("Required columns 'nlp_sentiment_score' or 'ticker' missing for sentiment enrichment.")
            return df

        df_enriched = df.copy()
        
        # Sort by ticker and datetime to ensure rolling windows work correctly
        df_enriched = df_enriched.sort_values(['ticker', 'datetime'])
        
        # Fill NaNs in nlp_sentiment_score with 0 (neutral) before calculations
        df_enriched['nlp_sentiment_score'] = df_enriched['nlp_sentiment_score'].fillna(0.0)

        # Process features per ticker to avoid data leakage between assets
        results = []
        for ticker, ticker_group in df_enriched.groupby('ticker'):
            ticker_group = ticker_group.copy()
            
            # 1. Rolling Statistics (Mean, Std, EMA)
            if 'rolling_mean' in self.enabled_features or 'rolling_std' in self.enabled_features:
                for window in self.windows:
                    if 'rolling_mean' in self.enabled_features:
                        ticker_group[f'sentiment_sma_{window}'] = ticker_group['nlp_sentiment_score'].rolling(window=window, min_periods=1).mean()
                    if 'rolling_std' in self.enabled_features:
                        ticker_group[f'sentiment_std_{window}'] = ticker_group['nlp_sentiment_score'].rolling(window=window, min_periods=1).std().fillna(0)
                
                ticker_group['sentiment_ema'] = ticker_group['nlp_sentiment_score'].ewm(span=self.windows[0], adjust=False).mean()

            # 2. Sentiment Velocity (Momentum)
            if 'velocity' in self.enabled_features:
                # Change over the last 3 intervals
                ticker_group['sentiment_velocity'] = ticker_group['nlp_sentiment_score'].diff(periods=3).fillna(0)

            # 3. News Intensity (Count of non-zero sentiment signals in window)
            if 'intensity' in self.enabled_features:
                # We assume a nlp_sentiment_score exists if it's not exactly 0 (or use a dedicated 'news_count' column if available)
                has_news = (ticker_group['nlp_sentiment_score'] != 0).astype(int)
                ticker_group['news_intensity'] = has_news.rolling(window=self.windows[0], min_periods=1).sum()

            # 4. Decay-Weighted Sentiment
            if 'decay_weighted' in self.enabled_features:
                ticker_group['sentiment_decay_weighted'] = self._calculate_decay_weights(ticker_group['nlp_sentiment_score'])

            results.append(ticker_group)

        # Recombine groups
        final_df = pd.concat(results).sort_index()
        
        logger.info(f"Sentiment enrichment complete. Added {len(final_df.columns) - len(df.columns)} features.")
        return final_df

    def _calculate_decay_weights(self, series: pd.Series) -> pd.Series:
        """
        Applies an exponential decay to historical sentiment scores.
        Recent scores have higher influence.
        """
        def apply_decay(x):
            weights = self.decay_factor ** np.arange(len(x))[::-1]
            return np.sum(x * weights) / np.sum(weights)

        # Apply using a rolling window equal to the shortest SMA window
        window_size = self.windows[0]
        return series.rolling(window=window_size, min_periods=1).apply(apply_decay, raw=True)