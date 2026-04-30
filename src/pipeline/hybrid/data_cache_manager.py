"""
Data Cache Manager: Handles cache validation and data freshness checks.
Extracted from HybridOrchestrator to improve code organization and testability.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.core.logging.logger import ProjectLogger


class DataCacheManager:
    """Manages data caching and validates cache freshness."""
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)
    
    def handle_data_caching(self, local_res: Dict[str, Any], force_training: bool,
                           batch_name: str, output_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Handle data caching logic and return features and targets dataframes."""
        n_f, n_t = local_res['results'].get('features_df'), local_res['results'].get('targets_df')
        if n_f is None or n_f.empty:
            self.logger.warning("⚠️ No features data in local results")
            return None, None
        
        # ✅ FIX: output_dir already includes batch_name!
        batch_dir = output_dir
        f_p = batch_dir / "features.parquet"
        t_p = batch_dir / "targets.parquet"
        
        has_cache = f_p.exists() and t_p.exists()
        
        if not has_cache or force_training:
            batch_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving new data to cache (force_training={force_training})")
            # Save the data
            n_f.to_parquet(f_p, compression='snappy')
            n_t.to_parquet(t_p, compression='snappy')
            self.logger.info(f"Data saved to: {f_p}, {t_p}")
            return n_f, n_t
        
        # Check if new data exists using cache comparison
        if not self._has_new_data(f_p, n_f):
            self.logger.info("⏭️ NO NEW DATA - Using existing cache.")
            try:
                n_f = pd.read_parquet(f_p)
                n_t = pd.read_parquet(t_p)
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading cache: {e}")
                return n_f, n_t
        else:
            # Save new data
            self.logger.info("New data found - updating cache.")
            n_f.to_parquet(f_p, compression='snappy')
            n_t.to_parquet(t_p, compression='snappy')
            self.logger.info(f"New data saved to: {f_p}, {t_p}")
        
        return n_f, n_t
    
    def _has_new_data(self, features_path: Path, new_features: pd.DataFrame) -> bool:
        """Check if new data exists compared to cache."""
        try:
            old_features = pd.read_parquet(features_path)
            
            # Create sets of (datetime, ticker) tuples to compare
            old_datetime = pd.to_datetime(old_features['datetime']).dt.tz_localize(None)
            new_datetime = pd.to_datetime(new_features['datetime']).dt.tz_localize(None)
            
            known = set(zip(old_datetime, old_features.get('ticker', [])))
            current = set(zip(new_datetime, new_features.get('ticker', [])))
            
            has_new = len(current - known) > 0
            
            if has_new:
                self.logger.info(f"📊 Found {len(current - known)} new data points")
            
            return has_new
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking cache freshness: {e}")
            # On error, assume new data exists
            return True
    
    def check_cache_status(self, cache_path: Path, new_data: pd.DataFrame) -> bool:
        """Check if cache needs updating."""
        if not cache_path.exists():
            self.logger.debug(f"💾 Cache miss: {cache_path}")
            return False
        
        return self._has_new_data(cache_path, new_data)
