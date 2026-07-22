# audit-ignore: ARCHITECTURAL_USAGE
"""
Data Batch Manager: Handles batch metadata creation, data merging, backups, and batch-level operations.
Extracted from HybridOrchestrator to improve code organization and testability.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

# Constants
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"
BATCH_METADATA_FILE = "batch_metadata.json"


class DataBatchManager:
    """Manages batch data operations including metadata creation, merging, and backups."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    def create_batch_metadata_dict(self, metadata: dict[str, Any], timestamp: str) -> dict[str, Any]:
        """Create batch metadata dictionary from run metadata."""
        return {
            'batch_name': metadata.get('batch_name'),
            'timestamp': timestamp,
            'tickers': metadata.get('tickers'),
            'timeframes': metadata.get('timeframes'),
            'heavy_models': metadata.get('heavy_models'),
            'files': metadata.get('saved_files', {})
        }

    def merge_with_existing(self, main_db_dir: Path, f_path: Path, t_path: Path,
                            features_df: pd.DataFrame, targets_df_filtered: pd.DataFrame, timestamp: str) -> None:
        """Merge new data with existing database."""
        try:
            e_f = pd.read_parquet(f_path)
            e_t = pd.read_parquet(t_path)
            feature_dedup_cols = self._dedup_columns(e_f, features_df)
            target_dedup_cols = self._dedup_columns(e_t, targets_df_filtered)
            accumulated_features = pd.concat([e_f, features_df]).drop_duplicates(
                subset=feature_dedup_cols, keep='last'
            )
            accumulated_targets = pd.concat([e_t, targets_df_filtered]).drop_duplicates(
                subset=target_dedup_cols, keep='last'
            )

            self._create_backup(main_db_dir, timestamp)
            self._save_dataframe(accumulated_features, f_path)
            self._save_dataframe(accumulated_targets, t_path)

            self.logger.info(f"✅ Data merged: {len(accumulated_features)} features, {len(accumulated_targets)} targets")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"❌ Error merging data: {e}")
            raise

    def create_new_database(self, main_db_dir: Path, features_df: pd.DataFrame,
                           targets_df_filtered: pd.DataFrame) -> None:
        """Create new database."""
        main_db_dir.mkdir(parents=True, exist_ok=True)
        self._save_dataframe(features_df, main_db_dir / FEATURES_FILE)
        self._save_dataframe(targets_df_filtered, main_db_dir / TARGETS_FILE)
        self.logger.info(f"✅ New database created: {main_db_dir}")

    def _create_backup(self, main_db_dir: Path, timestamp: str) -> None:
        """Create backup of main database."""
        backup_dir = Path("backups/accumulated") / f"main_database_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(main_db_dir, backup_dir / "main_database", dirs_exist_ok=True)
        self.logger.info(f"✅ Backup created: {backup_dir}")

    def save_batch_data(self, features_df: pd.DataFrame, targets_df_filtered: pd.DataFrame,
                       batch_dir: Path) -> None:
        """Save features and targets to batch directory."""
        f_out, t_out = batch_dir / FEATURES_FILE, batch_dir / TARGETS_FILE
        self._save_dataframe(features_df, f_out)
        self._save_dataframe(targets_df_filtered, t_out)
        self.logger.info(f"✅ Batch data saved to {batch_dir}")

    def save_heavy_config(self, batch_dir: Path, config: dict[str, Any]) -> None:
        """Save heavy configuration for Colab training."""
        config_path = batch_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=lambda x: self._serialize_config(x))
        self.logger.info(f"✅ Heavy config saved: {config_path}")

    def create_batch_metadata(self, batch_name: str, timestamp: str, tickers: list,
                             timeframes: list, features_df: pd.DataFrame,
                             targets_df_filtered: pd.DataFrame, batch_dir: Path,
                             heavy_models: list) -> dict[str, Any]:
        """Create batch metadata dictionary."""
        f_out, t_out = batch_dir / FEATURES_FILE, batch_dir / TARGETS_FILE
        return {
            'batch_name': batch_name,
            'timestamp': timestamp,
            'tickers': tickers,
            'timeframes': timeframes,
            'heavy_models': heavy_models,
            'features_shape': list(features_df.shape),
            'targets_shape': list(targets_df_filtered.shape),
            'files': {
                'features': str(f_out),
                'targets': str(t_out),
                'config': str(batch_dir / "config.json")
            }
        }

    def save_batch_metadata(self, batch_metadata: dict[str, Any], batch_dir: Path) -> None:
        """Save batch metadata to file."""
        metadata_path = batch_dir / BATCH_METADATA_FILE
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2, default=str)
        self.logger.info(f"✅ Batch metadata saved: {metadata_path}")

    def _save_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """Saves DataFrame to parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression='snappy', index=False)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"📝 Saved {len(df)} rows to {path.name}")

    def _dedup_columns(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> list | None:
        """Choose stable identity columns without collapsing separate timeframes."""
        candidates = ['datetime', 'ticker', 'interval']
        available = [col for col in candidates if col in existing_df.columns and col in new_df.columns]
        if len(available) >= 2:
            return available
        return available or None

    @staticmethod
    def _serialize_config(config: Any) -> Any:
        """Serialize configuration object."""
        if hasattr(config, 'as_dict'):
            return config.as_dict()
        return config
