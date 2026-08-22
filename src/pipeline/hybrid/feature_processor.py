# audit-ignore: ARCHITECTURAL_USAGE
# src/pipeline/hybrid/feature_processor.py
"""
Feature processing component for Hybrid Orchestrator.
Handles data normalization, feature/target splitting, and datetime processing.
"""

from pathlib import Path

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.target_column_utils import split_model_features_and_targets

logger = ProjectLogger.get_logger(__name__)


class FeatureProcessor:
    """Processes and normalizes features and targets data."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    def process_enriched_data(self, enriched_data) -> dict | None:
        """Process enriched data and return structured result."""
        if enriched_data is None or (isinstance(enriched_data, pd.DataFrame) and enriched_data.empty):
            self.logger.error("Stage 3 did not return enriched_data!")
            return None

        # Rebound on the very next line by a function that returns a new frame,
        # so the deep copy's data is discarded before anything reads it.
        enriched_df = enriched_data.copy(deep=False)
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
        """Preserve declared datetime semantics and normalize aware data to UTC."""
        if 'datetime' not in df.columns:
            return df
        from src.features.utils.datetime_utils import (
            ensure_datetime_column,
        )

        return ensure_datetime_column(df, raise_on_missing=True)

    def split_features_and_targets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into features and targets.

        CRITICAL: Targets DataFrame must contain ONLY target columns + minimal metadata
        to prevent data leakage.
        """
        feature_cols, target_cols, dropped_target_derived_cols = split_model_features_and_targets(df.columns)
        if dropped_target_derived_cols:
            self.logger.warning(
                "Dropped %s target-derived column(s) from features: %s",
                len(dropped_target_derived_cols),
                list(dropped_target_derived_cols)[:5],
            )

        features_df = df[feature_cols].copy()

        # CLEAN TARGETS: Only target columns + essential metadata
        essential_metadata = ['ticker', 'datetime', 'interval']
        targets_columns = target_cols + [col for col in essential_metadata if col in df.columns]
        targets_df = df[targets_columns].copy()

        self.logger.info(f"✅ Split: {len(feature_cols)} feature columns, {len(target_cols)} target columns")
        self.logger.info(f"   Targets DataFrame: {len(targets_df.columns)} columns (targets + metadata only)")

        return features_df, targets_df

    def get_datetime_column(self, df) -> str | None:
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
