# audit-ignore: ARCHITECTURAL_USAGE
"""
Data Processor - Handles data processing and normalization operations
"""

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class DataProcessor:
    """Handles data processing and normalization operations."""

    def __init__(self, data_utils):
        self.data_utils = data_utils
        self.logger = ProjectLogger.get_logger(__name__)

    def normalize_datetime_column(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Normalize datetime column name."""
        if datetime_col == 'published_at':
            df['datetime'] = df['published_at']
        return df

    def normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preserve declared datetime semantics and normalize aware data to UTC."""
        if 'datetime' not in df.columns:
            return df
        from src.features.utils.datetime_utils import (
            ensure_datetime_column,
        )

        return ensure_datetime_column(df, raise_on_missing=True)

    def split_features_and_targets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into features and targets."""
        target_cols = [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if c not in target_cols]

        features_df = df[feature_cols].copy()
        targets_df = df[target_cols].copy()

        # Ensure datetime column is in features
        datetime_col = self._get_datetime_column(df)
        if datetime_col and datetime_col not in features_df.columns:
            features_df[datetime_col] = df[datetime_col]

        return features_df, targets_df

    def get_datetime_column(self, df: pd.DataFrame) -> str | None:
        """Find datetime column."""
        if 'datetime' in df.columns:
            return 'datetime'
        elif 'published_at' in df.columns:
            return 'published_at'
        return None

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans DataFrame from NaN and Inf values."""
        return self.data_utils.clean_dataframe(df)

    def save_dataframe(self, df: pd.DataFrame, path) -> None:
        """Saves DataFrame to parquet."""
        self.data_utils.save_dataframe(df, path)

    def save_dataframes(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                       output_dir: str, batch_name: str) -> dict[str, Any]:
        """Save features and targets DataFrames."""
        from pathlib import Path

        batch_dir = Path(output_dir) / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        features_path = batch_dir / "features.parquet"
        targets_path = batch_dir / "targets.parquet"

        # Normalize datetime and timezone before saving
        features_df = self.normalize_datetime_column(features_df, 'datetime')
        features_df = self.normalize_timezone(features_df)

        self.save_dataframe(features_df, features_path)
        self.save_dataframe(targets_df, targets_path)

        return {
            'features_path': str(features_path),
            'targets_path': str(targets_path),
            'features_shape': features_df.shape,
            'targets_shape': targets_df.shape,
            'features_columns': list(features_df.columns),
            'targets_columns': list(targets_df.columns)
        }
