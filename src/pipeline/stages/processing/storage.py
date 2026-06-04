from datetime import datetime
from typing import Any

import pandas as pd

from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('ProcessingStorage')

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
                    path = f"data/processed/{key}_{timestamp}.parquet"
                    data.to_parquet(path)
                    saved_paths[key] = path
                elif isinstance(data, dict):
                    nested_paths = self._save_nested_dataframes(key, data, timestamp)
                    if nested_paths:
                        saved_paths[key] = nested_paths
            except Exception as e:
                self.logger.error(f"Error saving {key}: {e}")

        return saved_paths

    def _save_nested_dataframes(self, prefix: str, data: dict[str, Any], timestamp: str) -> dict[str, Any]:
        """Save DataFrame leaves in nested dictionaries."""
        saved_paths: dict[str, Any] = {}
        for key, value in data.items():
            safe_key = str(key).replace('/', '_').replace('\\', '_')
            nested_key = f"{prefix}_{safe_key}"
            if isinstance(value, pd.DataFrame):
                path = f"data/processed/{nested_key}_{timestamp}.parquet"
                value.to_parquet(path)
                saved_paths[key] = path
            elif isinstance(value, dict):
                child_paths = self._save_nested_dataframes(nested_key, value, timestamp)
                if child_paths:
                    saved_paths[key] = child_paths
        return saved_paths
