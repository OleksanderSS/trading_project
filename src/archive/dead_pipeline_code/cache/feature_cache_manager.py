import logging
import json
import hashlib
from typing import Dict, Any, Optional
import pandas as pd
from src.core.logging.logger import ProjectLogger
from src.features.feature_cache import get_feature_cache

logger = ProjectLogger.get_logger("FeatureEngineeringCacheManager")

class FeatureEngineeringCacheManager:
    def __init__(self, cache_dir: str):
        self.feature_cache = get_feature_cache(cache_dir=cache_dir)
        logger.info(f"✅ Feature cache initialized at {cache_dir}")

    def get_cached_features(self, ticker: str, timeframe: str, config: Any) -> Optional[pd.DataFrame]:
        cache_date_key = self._generate_cache_key(ticker, timeframe, config)
        config_hash = self._generate_config_hash(config)
        return self.feature_cache.get_features(ticker, cache_date_key, config_hash)

    def save_features(self, ticker: str, timeframe: str, config: Any, df: pd.DataFrame):
        cache_date_key = self._generate_cache_key(ticker, timeframe, config)
        config_hash = self._generate_config_hash(config)
        self.feature_cache.save_features(ticker, cache_date_key, config_hash, df)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"💾 Cached features for {ticker} {timeframe}")

    def _generate_cache_key(self, ticker: str, timeframe: str, config: Any) -> str:
        config_hash = self._generate_config_hash(config)
        return f"{ticker}_{timeframe}_{config_hash[:8]}"

    def _generate_config_hash(self, config: Any) -> str:
        # Assuming config has get_config_hash or is dict-like
        config_data = config.get_config_hash() if hasattr(config, "get_config_hash") else str(config)
        return hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
