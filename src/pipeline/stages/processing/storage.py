from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('ProcessingStorage')

PERSISTENT_MACRO_PATH = Path("data/processed/features/macro_data.parquet")

class ProcessingStorage:
    """Handles saving and loading of processed data."""

    def __init__(self, file_manager: FileManager):
        self.logger = logger
        self.file_manager = file_manager

    def save_cleaned_data_to_files(self, filtered_results: dict[str, Any]) -> dict[str, Any]:
        """Save results to local storage."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_paths = {}

        for key, data in filtered_results.items():
            try:
                if isinstance(data, pd.DataFrame):
                    if 'published_at' in data.columns:
                        data['published_at'] = pd.to_datetime(data['published_at'], utc=True, errors='coerce')
                    path = f"data/processed/{key}_{timestamp}.parquet"
                    data.to_parquet(path)
                    saved_paths[key] = path
                    if key == "macro_data":
                        persistent = self._save_persistent_macro_snapshot(data)
                        saved_paths["macro_data_persistent"] = persistent
                elif isinstance(data, dict):
                    nested_paths = self._save_nested_dataframes(key, data, timestamp)
                    if nested_paths:
                        saved_paths[key] = nested_paths
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f"Error saving {key}: {e}")

        return saved_paths

    def _save_persistent_macro_snapshot(self, data: pd.DataFrame) -> str:
        """Atomically persist only point-in-time-safe macro observations."""
        required = {"datetime", "series_id", "value"}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(
                "Persistent macro snapshot missing canonical columns: "
                + ", ".join(missing)
            )
        availability = next(
            (
                column
                for column in ("available_at", "released_at", "realtime_start")
                if column in data.columns
            ),
            None,
        )
        if availability is None:
            raise ValueError(
                "Persistent macro snapshot requires available_at, released_at, "
                "or realtime_start."
            )
        if data.empty or pd.to_datetime(
            data[availability], errors="coerce", utc=True
        ).isna().any():
            raise ValueError(
                "Persistent macro snapshot cannot contain empty or invalid "
                f"{availability} values."
            )
        target_path = self.file_manager.base_dir / PERSISTENT_MACRO_PATH
        # This path is a persistent, accumulating history (read by
        # cli/pipeline_data_loader.py and cli/pipeline_executor.py as the
        # "persistent fallback" macro dataset), but `data` here is only
        # this run's incremental delta (collectors like FredCollector
        # filter to new records only). A plain overwrite would silently
        # destroy every prior run's accumulated history down to just the
        # latest batch - merge with whatever's already on disk instead,
        # keeping the newest row per (series_id, datetime) so revisions
        # to an already-seen observation still take effect.
        existing = self.file_manager.load_dataframe(target_path, format="parquet")
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["series_id", "datetime"], keep="last")
            combined = combined.sort_values(["series_id", "datetime"]).reset_index(drop=True)
        else:
            combined = data
        self.file_manager.save_dataframe(
            combined,
            target_path,
            format="parquet",
            remove_tz=False,
            index=False,
        )
        return str(PERSISTENT_MACRO_PATH)

    def _save_nested_dataframes(self, prefix: str, data: dict[str, Any], timestamp: str) -> dict[str, Any]:
        """Save DataFrame leaves in nested dictionaries."""
        saved_paths: dict[str, Any] = {}
        for key, value in data.items():
            safe_key = str(key).replace('/', '_').replace('\\', '_')
            nested_key = f"{prefix}_{safe_key}"
            if isinstance(value, pd.DataFrame):
                if 'published_at' in value.columns:
                    value['published_at'] = pd.to_datetime(value['published_at'], utc=True, errors='coerce')
                path = f"data/processed/{nested_key}_{timestamp}.parquet"
                value.to_parquet(path)
                saved_paths[key] = path
            elif isinstance(value, dict):
                child_paths = self._save_nested_dataframes(nested_key, value, timestamp)
                if child_paths:
                    saved_paths[key] = child_paths
        return saved_paths
