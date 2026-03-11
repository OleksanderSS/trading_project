import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from src.config.unified_config_manager import get_current_config
from src.features.enrichers.base import BaseEnricher
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("MacroFeaturesEnricher")

class MacroFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with macroeconomic indicators from FRED.
    Implements caching to avoid repeated downloads.
    """
    
    @property
    def name(self) -> str:
        return "macro_features"

    def __init__(self, cache_dir: str = './cache'):
        self.config = get_current_config().get('macro_features.macro_fred_series', {})
        self.cache_path = Path(cache_dir) / 'macro_data.parquet'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config:
            logger.warning("Configuration for macro features ('macro_fred_series') not found.")

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds macro features to the DataFrame.

        Args:
            df: DataFrame with a DatetimeIndex.

        Returns:
            DataFrame with added macro features.
        """
        if not self.config or df.empty:
            return df

        start_date = df.index.min()
        end_date = df.index.max()

        macro_data = self._load_macro_data(start_date, end_date)

        if macro_data.empty:
            logger.warning("Could not load macro data. Skipping enrichment.")
            return df

        logger.info("Joining macro data with the main DataFrame...")
        aligned_macro_data = macro_data.reindex(df.index, method='ffill')
        
        df_enriched = df.join(aligned_macro_data, how='left')
        
        # Use the non-deprecated bfill() method instead of fillna(method='bfill')
        df_enriched = df_enriched.bfill()

        logger.info("Macro features successfully added.")
        return df_enriched

    def _load_macro_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if self._is_cache_valid(start_date, end_date):
            logger.info(f"Loading macro data from cache: {self.cache_path}")
            return pd.read_parquet(self.cache_path)

        logger.info("Cache not found or outdated. Loading data from FRED...")
        series_ids = list(self.config.values())
        series_names = list(self.config.keys())

        try:
            fred_data = web.DataReader(series_ids, 'fred', start_date, end_date)
            fred_data.columns = series_names
            
            fred_data.to_parquet(self.cache_path)
            logger.info(f"Macro data saved to cache: {self.cache_path}")
            return fred_data
        except Exception as e:
            logger.error(f"Error loading data from FRED: {e}", exc_info=True)
            return pd.DataFrame()

    def _is_cache_valid(self, start_date: datetime, end_date: datetime) -> bool:
        if not self.cache_path.exists():
            return False
        
        try:
            cached_df = pd.read_parquet(self.cache_path)
            if cached_df.index.min() <= start_date and cached_df.index.max() >= end_date:
                logger.info("Cache fully covers the required date range.")
                return True
            else:
                logger.info("Date range in cache is insufficient. Refresh required.")
                return False
        except Exception as e:
            logger.warning(f"Error reading cache file {self.cache_path}: {e}. A reload will be performed.")
            return False