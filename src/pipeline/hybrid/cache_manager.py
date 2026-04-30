# src/pipeline/hybrid/cache_manager.py
"""
Cache management component for Hybrid Orchestrator.
Handles feature selection caching and data change detection.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Constants
FEATURES_FILE = "features.parquet"


class CacheManager:
    """Manages caching operations for hybrid pipeline."""
    
    def __init__(self, output_dir: Path, batch_name: str):
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
    
    def check_if_feature_selection_needed(self, batch_dir: Path, new_rows_count: int, force: bool = False) -> Dict[str, Any]:
        """Checks if new feature selection is required."""
        # Check for initial run or forced selection
        initial_check = self._check_initial_or_forced_selection(batch_dir, force)
        if initial_check:
            return initial_check
            
        # Check time-based selection
        time_check = self._check_time_based_selection(batch_dir)
        if time_check:
            return time_check
            
        # Check data change percentage
        return self._check_data_change_percentage(batch_dir, new_rows_count)
    
    def _check_initial_or_forced_selection(self, batch_dir: Path, force: bool) -> Optional[Dict[str, Any]]:
        """Check if selection is needed due to initial run or force."""
        files = list(batch_dir.glob("selected_features_*.json"))
        if not files:
            return {'needed': True, 'reason': 'No selected_features files found (initial run)'}
        if force:
            return {'needed': True, 'reason': 'Forced feature selection requested'}
        return None
    
    def _check_time_based_selection(self, batch_dir: Path) -> Optional[Dict[str, Any]]:
        """Check if selection is needed based on time elapsed."""
        files = list(batch_dir.glob("selected_features_*.json"))
        if not files:
            return None
            
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                last_ts = data.get('timestamp')
            if last_ts:
                days = (datetime.now() - datetime.fromisoformat(last_ts)).days
                if days > 7:
                    return {'needed': True, 'reason': f'{days} days passed (> 7 days)'}
        except Exception:
            pass
        return None
    
    def _check_data_change_percentage(self, batch_dir: Path, new_rows_count: int) -> Dict[str, Any]:
        """Check if selection is needed based on data change percentage."""
        f_path = batch_dir / FEATURES_FILE
        if not f_path.exists():
            return {'needed': True, 'reason': 'No features file found'}
            
        try:
            old_rows = len(pd.read_parquet(f_path))
            if old_rows > 0:
                pct = (new_rows_count / old_rows) * 100
                if pct > 10:
                    return {'needed': True, 'reason': f'Data changed by {pct:.1f}% (> 10%)'}
        except Exception:
            pass
            
        return {'needed': False, 'reason': 'No significant changes detected'}
    
    def check_cache_status(self, features_path: Path, new_features: pd.DataFrame) -> bool:
        """Check if cache has new data."""
        try:
            if not features_path.exists():
                return True
                
            old_features = pd.read_parquet(features_path)
            known = set(zip(pd.to_datetime(old_features['datetime']).dt.tz_localize(None), old_features['ticker']))
            current = set(zip(pd.to_datetime(new_features['datetime']).dt.tz_localize(None), new_features['ticker']))
            return len(current - known) > 0
        except Exception:
            return True
