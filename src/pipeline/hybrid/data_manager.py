"""
Data management component for Hybrid Orchestrator.
Handles data loading, saving, and processing operations.
"""
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)
FEATURES_FILE = 'features.parquet'
TARGETS_FILE = 'targets.parquet'


class HybridDataManager:
    """Manages data operations for hybrid pipeline."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    def load_features_data(self) ->pd.DataFrame:
        """Load features data from parquet file."""
        features_path = self.output_dir / FEATURES_FILE
        if not features_path.exists():
            self.logger.warning(f'Features file not found: {features_path}')
            return pd.DataFrame()
        try:
            features_df = pd.read_parquet(features_path)
            self.logger.info(f'Loaded features: {features_df.shape}')
            return features_df
        except Exception as e:
            self.logger.error(
                f'Error loading features from {features_path}: {e}', exc_info=True)
            return pd.DataFrame()

    def load_targets_data(self) ->pd.DataFrame:
        """Load targets data from parquet file."""
        targets_path = self.output_dir / TARGETS_FILE
        if not targets_path.exists():
            self.logger.warning(f'Targets file not found: {targets_path}')
            return pd.DataFrame()
        try:
            targets_df = pd.read_parquet(targets_path)
            self.logger.info(f'Loaded targets: {targets_df.shape}')
            return targets_df
        except Exception as e:
            self.logger.error(f'Error loading targets from {targets_path}: {e}', exc_info=True
                )
            return pd.DataFrame()

    def save_dataframes(self, features_df: pd.DataFrame, targets_df: pd.
        DataFrame) ->dict[str, Any]:
        """Save features and targets DataFrames."""
        batch_dir = self.output_dir
        batch_dir.mkdir(parents=True, exist_ok=True)
        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        self.save_dataframe(features_df, features_path)
        self.save_dataframe(targets_df, targets_path)
        self.logger.info(f'Features saved: {features_path}')
        self.logger.info(f'Targets saved: {targets_path}')
        return {'paths': {'features': str(features_path), 'targets': str(
            targets_path)}}

    def save_dataframe(self, df: pd.DataFrame, path: Path):
        """Saves DataFrame to parquet."""
        if df is None or df.empty:
            return
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(path, compression='snappy')

    def clean_dataframe(self, df: pd.DataFrame) ->pd.DataFrame:
        """Cleans DataFrame from NaN and Inf values."""
        if df is None or df.empty:
            return df
        df = df.copy()
        import numpy as np
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df

    def has_new_data(self, features_path: Path, new_features: pd.DataFrame
        ) ->bool:
        """Check if new data exists compared to cache."""
        try:
            old_features = pd.read_parquet(features_path)
            known = set(zip(pd.to_datetime(old_features['datetime']).dt.
                tz_localize(None), old_features['ticker'], strict=False))
            current = set(zip(pd.to_datetime(new_features['datetime']).dt.
                tz_localize(None), new_features['ticker'], strict=False))
            return len(current - known) > 0
        except Exception as e:
            self.logger.error(f'Error checking for new data: {e}', exc_info=True)
            return True
