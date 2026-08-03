from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.feature_orchestrator import FeatureOrchestrator

logger = ProjectLogger.get_logger('FeatureEnricher')

class FeatureEnricher:
    """Handles feature generation and enrichment."""

    def __init__(self, config_manager: Any):
        self.logger = logger
        self.orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        # A FeatureCache was constructed here and never consulted -- no call
        # to get_features or save_features existed anywhere. Moved to
        # src/archive/features_superseded/ on 2026-08-02.
        #
        # It could not have been wired as written. Three reasons, in
        # increasing order of seriousness:
        #
        # 1. Its API is get_features(ticker, date, config_hash), and NOTHING
        #    in the project computes a config_hash. The entire invalidation
        #    story rested on an argument no caller produces.
        # 2. The granularity is wrong. enrich_features receives a whole
        #    multi-ticker frame for one timeframe; the cache is keyed per
        #    ticker AND date.
        # 3. That granularity is not merely inconvenient, it is unsound. 14
        #    of the 20 enrichers use rolling windows or groupby-shift, so a
        #    row's features depend on its neighbours. Caching per (ticker,
        #    date) and reassembling would corrupt every rolling computation
        #    at the seams -- silently, since the output would still be a
        #    full-looking frame of plausible numbers.
        #
        # The job it claimed is already done properly one level up:
        # PipelineExecutor reuses features.parquet/targets.parquet when the
        # raw-data fingerprint AND the code fingerprint both match. That is
        # whole-frame granularity (so rolling windows stay intact) with
        # invalidation that actually exists.

    def enrich_features(self, df: pd.DataFrame, timeframe: str, **kwargs) -> pd.DataFrame:
        """Enrich data with technical and statistical features."""
        self.logger.info(f"Enriching features for timeframe: {timeframe}")
        # ✅ FIX: pass kwargs (macro_data, news) to orchestrator.run so enrichers receive them
        return self.orchestrator.run(df, add_timeframe_suffix=True, timeframe=timeframe, **kwargs)

    def add_macro_features(self, df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Add macro-economic indicators."""
        return df # Integration logic here
