# src/pipeline/hybrid/feature_processor.py
"""
Feature processing component for Hybrid Orchestrator.
Handles data normalization, feature/target splitting, and datetime processing.
"""

import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class FeatureProcessor:
    """Processes and normalizes features and targets data."""
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
    
    def process_enriched_data(self, enriched_data) -> Optional[dict]:
        """Process enriched data and return structured result."""
        if enriched_data is None or (isinstance(enriched_data, pd.DataFrame) and enriched_data.empty):
            self.logger.error("Stage 3 did not return enriched_data!")
            return None
            
        enriched_df = enriched_data.copy()
        enriched_df = self.normalize_datetime_index(enriched_df)
        datetime_col = self.get_datetime_column(enriched_df)
        
        if datetime_col is not None:
            enriched_df = self.normalize_datetime_column(enriched_df, datetime_col)
            enriched_df = self.normalize_timezone(enriched_df)
        else:
            self.logger.warning("Datetime column not found — proceeding without datetime normalization")
        
        # Split features and targets
        features_df, targets_df = self.split_features_and_targets(enriched_df)
        
        return {
            'data': enriched_df,
            'features': features_df,
            'targets': targets_df
        }
    
    def normalize_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime index to column."""
        if df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
            return df.reset_index()
        return df
    
    def normalize_datetime_column(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Normalize datetime column name."""
        if datetime_col == 'published_at':
            df['datetime'] = df['published_at']
        return df
    
    def normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timezone for datetime column."""
        if 'datetime' in df.columns:
            tmp_dt = pd.to_datetime(df['datetime'])
            df['datetime'] = tmp_dt.dt.tz_localize(None) if tmp_dt.dt.tz is not None else tmp_dt
        return df
    
    def split_features_and_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into features and targets.
        
        CRITICAL: Targets DataFrame must contain ONLY target columns + minimal metadata
        to prevent data leakage.
        """
        target_cols = [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if c not in target_cols]
        
        features_df = df[feature_cols].copy()
        
        # CLEAN TARGETS: Only target columns + essential metadata
        essential_metadata = ['ticker', 'datetime', 'interval']
        targets_columns = target_cols + [col for col in essential_metadata if col in df.columns]
        targets_df = df[targets_columns].copy()
        
        self.logger.info(f"✅ Split: {len(feature_cols)} feature columns, {len(target_cols)} target columns")
        self.logger.info(f"   Targets DataFrame: {len(targets_df.columns)} columns (targets + metadata only)")
        
        return features_df, targets_df
    
    def get_datetime_column(self, df) -> Optional[str]:
        """Find datetime column."""
        if 'datetime' in df.columns:
            return 'datetime'
        elif 'published_at' in df.columns:
            return 'published_at'
        else:
            return None
    
    def save_enriched_data(self, processed_data: dict, batch_dir: Path) -> dict:
        """Save enriched data to files."""
        features_df = processed_data['features']
        targets_df = processed_data['targets']
        
        features_path = batch_dir / "features.parquet"
        targets_path = batch_dir / "targets.parquet"
        
        # Save DataFrames
        if not features_df.empty:
            features_df.to_parquet(features_path, compression='snappy')
            self.logger.info(f"Features saved to: {features_path}")
        
        if not targets_df.empty:
            targets_df.to_parquet(targets_path, compression='snappy')
            self.logger.info(f"Targets saved to: {targets_path}")
        
        return {
            'features_path': features_path,
            'targets_path': targets_path
        }
