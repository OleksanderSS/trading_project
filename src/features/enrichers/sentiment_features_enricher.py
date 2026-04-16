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
        Uses news data from kwargs if available.

        Args:
            df: DataFrame containing at least 'datetime', 'ticker'.
            **kwargs: May contain 'news' DataFrame with sentiment scores.

        Returns:
            DataFrame with additional sentiment features.
        """
        if df.empty:
            logger.warning("Received an empty DataFrame for sentiment enrichment.")
            return df

        if 'ticker' not in df.columns:
            logger.error("Required column 'ticker' missing for sentiment enrichment.")
            return df

        # ✅ Спробуємо отримати sentiment з різних джерел
        sentiment_col = None
        if 'nlp_sentiment_score' in df.columns:
            sentiment_col = 'nlp_sentiment_score'
        elif 'sentiment_score' in df.columns:
            sentiment_col = 'sentiment_score'
        elif 'sentiment' in df.columns:
            sentiment_col = 'sentiment'
        
        # ✅ Якщо немає sentiment в df, спробуємо взяти з news в kwargs
        if sentiment_col is None:
            news_df = kwargs.get('news')
            if news_df is not None and isinstance(news_df, pd.DataFrame) and not news_df.empty:
                logger.info(f"Attempting to merge sentiment from news data ({len(news_df)} rows)")
                # Шукаємо колонку часу в news
                time_col = None
                possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
                for col in possible_time_cols:
                    if col in news_df.columns:
                        time_col = col
                        break
                        
                news_sentiment_col = None
                for col in ['sentiment_score', 'sentiment', 'finbert_score']:
                    if col in news_df.columns:
                        news_sentiment_col = col
                        break
                
                if news_sentiment_col and time_col:
                    # ✅ FIX: Normalize timezone and convert to datetime64[ns]
                    news_df[time_col] = pd.to_datetime(news_df[time_col], errors='coerce', utc=True)
                    if news_df[time_col].dt.tz is not None:
                        news_df[time_col] = news_df[time_col].dt.tz_localize(None)
                    news_df[time_col] = news_df[time_col].astype('datetime64[ns]')
                    
                    logger.info(f"✅ Found time column '{time_col}' with {len(news_df[news_df[time_col].notna()])} valid timestamps")
                    
                    # Агрегуємо sentiment по даті та тікеру (якщо є)
                    if 'ticker' in news_df.columns:
                        sentiment_agg = news_df.groupby(['ticker', pd.Grouper(key=time_col, freq='1h')])[news_sentiment_col].mean().reset_index()
                        sentiment_agg.columns = ['ticker', 'datetime', 'nlp_sentiment_score']
                    else:
                        # Якщо немає тікера в новинах, агрегуємо тільки по даті (глобальний sentiment)
                        sentiment_agg = news_df.groupby(pd.Grouper(key=time_col, freq='1h'))[news_sentiment_col].mean().reset_index()
                        sentiment_agg.columns = ['datetime', 'nlp_sentiment_score']
                    
                    # ✅ Нормалізуємо timezone + precision в sentiment_agg
                    if pd.api.types.is_datetime64_any_dtype(sentiment_agg['datetime']):
                        if hasattr(sentiment_agg['datetime'].dtype, 'tz') and sentiment_agg['datetime'].dt.tz is not None:
                            sentiment_agg['datetime'] = sentiment_agg['datetime'].dt.tz_localize(None)
                        # Convert to ns precision
                        if sentiment_agg['datetime'].dtype != 'datetime64[ns]':
                            sentiment_agg['datetime'] = sentiment_agg['datetime'].astype('datetime64[ns]')
                    
                    # Merge з основним df
                    has_datetime = 'datetime' in df.columns or df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex)
                    
                    if has_datetime:
                        if 'datetime' not in df.columns:
                            df = df.reset_index(names='datetime' if not df.index.name else None)
                        
                        # ✅ Нормалізуємо timezone + precision в df
                        if pd.api.types.is_datetime64_any_dtype(df['datetime']):
                            if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
                                df['datetime'] = df['datetime'].dt.tz_localize(None)
                            # Convert to ns precision
                            if df['datetime'].dtype != 'datetime64[ns]':
                                df['datetime'] = df['datetime'].astype('datetime64[ns]')
                            
                        merge_keys = ['ticker', 'datetime'] if 'ticker' in sentiment_agg.columns else ['datetime']
                        df = df.merge(sentiment_agg, on=merge_keys, how='left')
                        df = df.set_index('datetime')
                        sentiment_col = 'nlp_sentiment_score'
                        logger.info(f"✅ Merged sentiment from news data")
                else:
                    logger.warning(f"Missing required columns: news_sentiment_col={news_sentiment_col}, time_col={time_col}")
        
        if sentiment_col is None:
            logger.warning("No sentiment data available. Skipping sentiment enrichment.")
            return df

        df_enriched = df.copy()
        
        # Sort by ticker and datetime to ensure rolling windows work correctly
        if 'datetime' in df_enriched.columns:
            df_enriched = df_enriched.sort_values(['ticker', 'datetime'])
        
        # Fill NaNs in sentiment with forward fill first, then backward fill, then 0 (neutral)
        # This preserves the last known sentiment rather than assuming neutral
        df_enriched[sentiment_col] = df_enriched.groupby('ticker')[sentiment_col].ffill().bfill().fillna(0.0)

        # Process features per ticker to avoid data leakage between assets
        results = []
        for ticker, ticker_group in df_enriched.groupby('ticker'):
            ticker_group = ticker_group.copy()
            
            # 1. Rolling Statistics (Mean, Std, EMA)
            if 'rolling_mean' in self.enabled_features or 'rolling_std' in self.enabled_features:
                for window in self.windows:
                    if 'rolling_mean' in self.enabled_features:
                        ticker_group[f'sentiment_sma_{window}'] = ticker_group[sentiment_col].rolling(window=window, min_periods=1).mean()
                    if 'rolling_std' in self.enabled_features:
                        ticker_group[f'sentiment_std_{window}'] = ticker_group[sentiment_col].rolling(window=window, min_periods=1).std().fillna(0)
                
                ticker_group['sentiment_ema'] = ticker_group[sentiment_col].ewm(span=self.windows[0], adjust=False).mean()

            # 2. Sentiment Velocity (Momentum)
            if 'velocity' in self.enabled_features:
                # Change over the last 3 intervals
                ticker_group['sentiment_velocity'] = ticker_group[sentiment_col].diff(periods=3).fillna(0)

            # 3. News Intensity (Count of non-zero sentiment signals in window)
            if 'intensity' in self.enabled_features:
                has_news = (ticker_group[sentiment_col] != 0).astype(int)
                ticker_group['news_intensity'] = has_news.rolling(window=self.windows[0], min_periods=1).sum()

            # 4. Decay-Weighted Sentiment
            if 'decay_weighted' in self.enabled_features:
                ticker_group['sentiment_decay_weighted'] = self._calculate_decay_weights(ticker_group[sentiment_col])

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