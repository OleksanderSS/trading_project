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

    @property
    def name(self) -> str:
        """Unique identifier for the enricher."""
        return "hype_scores"

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates hype scores using a left-closed rolling window to prevent look-ahead bias.
        
        Args:
            df: Input DataFrame containing at least 'ticker' and 'published_at' columns.
            **kwargs: 
                hype_window (str): The rolling window size (e.g., '1h', '24h'). Defaults to '1h'.
        
        Returns:
            DataFrame with an additional 'hype_score' column.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping hype calculation.")
            return df

        if 'ticker' not in df.columns or 'published_at' not in df.columns:
            logger.error("Required columns 'ticker' or 'published_at' not found for hype enrichment.")
            return df

        hype_window = kwargs.get('hype_window', '1h')
        logger.info(f"Calculating hype scores using window: {hype_window}")

        try:
            df_enriched = df.copy()
            
            # Ensure published_at is datetime and sorted for rolling operations
            df_enriched['published_at'] = pd.to_datetime(df_enriched['published_at'])
            df_enriched = df_enriched.sort_values('published_at')
            
            # Calculate rolling count per ticker
            # 'closed=left' is critical to ensure we only count past events (no leakage)
            df_enriched['hype_score'] = df_enriched.groupby('ticker')['published_at'].transform(
                lambda x: x.rolling(hype_window, closed='left').count()
            ).fillna(0)

            logger.info("Hype scores enrichment completed successfully.")
            return df_enriched

        except Exception as e:
            logger.error(f"Error during hype enrichment: {e}", exc_info=True)
            return df