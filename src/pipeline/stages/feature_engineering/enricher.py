from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.feature_cache import get_feature_cache
from src.features.feature_orchestrator import FeatureOrchestrator

logger = ProjectLogger.get_logger('FeatureEnricher')

class FeatureEnricher:
    """Handles feature generation and enrichment."""

    def __init__(self, config_manager: Any):
        self.logger = logger
        self.orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        cache_dir = config_manager.get('performance.feature_cache_dir', 'data/cache/features')
        self.feature_cache = get_feature_cache(cache_dir=cache_dir)

    def enrich_features(self, df: pd.DataFrame, timeframe: str, **kwargs) -> pd.DataFrame:
        """Enrich data with technical and statistical features."""
        self.logger.info(f"Enriching features for timeframe: {timeframe}")
        # ✅ FIX: pass kwargs (macro_data, news) to orchestrator.run so enrichers receive them
        return self.orchestrator.run(df, add_timeframe_suffix=True, timeframe=timeframe, **kwargs)

    def add_macro_features(self, df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Add macro-economic indicators."""
        return df # Integration logic here
