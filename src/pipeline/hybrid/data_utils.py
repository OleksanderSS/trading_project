# audit-ignore: ARCHITECTURAL_USAGE
"""
Data Utilities for Hybrid Orchestrator.
Handles data cleaning, validation, and utility functions.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class DataUtils:
    """Utility functions for data operations."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by handling NaN and infinite values."""
        if df.empty:
            return df

        df = df.copy()

        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        return df

    def save_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """Saves DataFrame to parquet."""
        if df is None or df.empty:
            return

        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(path, compression='snappy')

    def get_datetime_column(self, df: pd.DataFrame) -> str | None:
        """Find datetime column."""
        if 'datetime' in df.columns:
            return 'datetime'
        elif 'published_at' in df.columns:
            return 'published_at'
        return None

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

    def split_features_and_targets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into features and targets."""
        target_cols = [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if c not in target_cols]

        features_df = df[feature_cols].copy()
        targets_df = df[target_cols].copy()

        # Copy essential columns to features
        for col in ['ticker', 'datetime']:
            if col in df.columns:
                features_df[col] = df[col]

        return features_df, targets_df

    def save_dataframes(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                        batch_dir: Path) -> dict[str, Any]:
        """Save features and targets DataFrames."""
        batch_dir.mkdir(parents=True, exist_ok=True)

        features_path = batch_dir / 'features.parquet'
        targets_path = batch_dir / 'targets.parquet'

        self.save_dataframe(features_df, features_path)
        self.save_dataframe(targets_df, targets_path)

        return {
            'paths': {
                'features': str(features_path),
                'targets': str(targets_path)
            },
            'shapes': {
                'features': features_df.shape,
                'targets': targets_df.shape
            }
        }
