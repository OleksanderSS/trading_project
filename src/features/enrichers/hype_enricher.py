import pandas as pd
import logging
from typing import Dict, Any, Optional
from src.features.enrichers.base import BaseEnricher
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("HypeEnricher")

class HypeEnricher(BaseEnricher):
    """
    Enriches the DataFrame with hype scores by counting news occurrences 
    within a rolling time window to gauge market attention.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config from FeatureOrchestrator."""
        self.config = config or {}

    @property
    def name(self) -> str:
        """Unique identifier for the enricher."""
        return "hype_features"
    
    @property
    def priority(self) -> int:
        """Execution order - run after NLP (30) and sentiment (40)"""
        return 50

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates hype scores using news data from kwargs.
        
        Args:
            df: Input DataFrame (market data)
            **kwargs: 
                news: DataFrame with news data
                hype_window (str): The rolling window size (e.g., '1h', '24h'). Defaults to '1h'.
        
        Returns:
            DataFrame with an additional 'hype_score' column.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping hype calculation.")
            return df

        # ✅ Отримуємо новини з kwargs
        news_df = kwargs.get('news')
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            logger.warning("No news data available in kwargs. Skipping hype enrichment.")
            return df

        # Шукаємо колонку часу в новинах
        time_col = None
        possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
        for col in possible_time_cols:
            if col in news_df.columns:
                time_col = col
                break
        
        if time_col is None:
            logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping hype enrichment.")
            return df

        hype_window = kwargs.get('hype_window', '1h')
        logger.info(f"Calculating hype scores using window: {hype_window} from {len(news_df)} news items")

        try:
            df_enriched = df.copy()
            news_copy = news_df.copy()
            
            # ✅ FIX: Normalize timezone and convert to datetime64[ns]
            news_copy[time_col] = pd.to_datetime(news_copy[time_col], errors='coerce', utc=True)
            if news_copy[time_col].dt.tz is not None:
                news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
            news_copy[time_col] = news_copy[time_col].astype('datetime64[ns]')
            news_copy = news_copy.dropna(subset=[time_col]).sort_values(time_col)
            
            logger.info(f"✅ Found time column '{time_col}' with {len(news_copy)} valid timestamps")
            
            # Normalize timezone in df_enriched datetime column
            if 'datetime' in df_enriched.columns:
                if pd.api.types.is_datetime64_any_dtype(df_enriched['datetime']):
                    if hasattr(df_enriched['datetime'].dtype, 'tz') and df_enriched['datetime'].dt.tz is not None:
                        df_enriched['datetime'] = df_enriched['datetime'].dt.tz_localize(None)
                    # ✅ Convert to ns precision
                    if df_enriched['datetime'].dtype != 'datetime64[ns]':
                        df_enriched['datetime'] = df_enriched['datetime'].astype('datetime64[ns]')
            
            # Рахуємо hype по тікеру та часу
            if 'ticker' in news_copy.columns and 'ticker' in df_enriched.columns:
                # Агрегуємо кількість новин по тікеру та часу
                news_count = news_copy.groupby(['ticker', pd.Grouper(key=time_col, freq='1h')]).size().reset_index(name='news_count')
                news_count = news_count.rename(columns={time_col: 'datetime'})
                
                # ✅ Нормалізуємо timezone + precision в news_count
                if 'datetime' in news_count.columns:
                    if pd.api.types.is_datetime64_any_dtype(news_count['datetime']):
                        if hasattr(news_count['datetime'].dtype, 'tz') and news_count['datetime'].dt.tz is not None:
                            news_count['datetime'] = news_count['datetime'].dt.tz_localize(None)
                        # Convert to ns precision
                        if news_count['datetime'].dtype != 'datetime64[ns]':
                            news_count['datetime'] = news_count['datetime'].astype('datetime64[ns]')
                
                # Merge з основним df
                if 'datetime' in df_enriched.columns:
                    df_enriched = df_enriched.merge(news_count, on=['ticker', 'datetime'], how='left')
                    df_enriched['hype_score'] = df_enriched['news_count'].fillna(0)
                    df_enriched = df_enriched.drop(columns=['news_count'])
                    logger.info(f"✅ Added hype_score based on news count per ticker")
            else:
                # Глобальний hype (без тікера)
                news_count = news_copy.groupby(pd.Grouper(key=time_col, freq='1h')).size().reset_index(name='news_count')
                news_count = news_count.rename(columns={time_col: 'datetime'})
                
                # ✅ Нормалізуємо timezone + precision в news_count
                if 'datetime' in news_count.columns:
                    if pd.api.types.is_datetime64_any_dtype(news_count['datetime']):
                        if hasattr(news_count['datetime'].dtype, 'tz') and news_count['datetime'].dt.tz is not None:
                            news_count['datetime'] = news_count['datetime'].dt.tz_localize(None)
                        # Convert to ns precision
                        if news_count['datetime'].dtype != 'datetime64[ns]':
                            news_count['datetime'] = news_count['datetime'].astype('datetime64[ns]')
                
                if 'datetime' in df_enriched.columns:
                    df_enriched = df_enriched.merge(news_count, on='datetime', how='left')
                    df_enriched['hype_score'] = df_enriched['news_count'].fillna(0)
                    df_enriched = df_enriched.drop(columns=['news_count'])
                    logger.info(f"✅ Added global hype_score based on news count")

            logger.info("Hype scores enrichment completed successfully.")
            return df_enriched

        except Exception as e:
            logger.error(f"Error during hype enrichment: {e}", exc_info=True)
            return df