import asyncio
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.security.path_validator import PathValidationError, validate_safe_path
from src.data.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)


class LocalFileCollector(BaseCollector):
    """
    A collector that reads data from local files (CSV or Parquet) and
    integrates it into the standard collection pipeline.
    """
    collector_type = 'local_file'

    def __init__(self, configs: dict[str, Any], http_client_factory,
        db_manager, cache_manager=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)
        raw_path = self.configs.get('file_path')
        self.file_type = self.configs.get('file_type', 'csv').lower()
        self.date_col = self.configs.get('date_col')

        if not raw_path:
            self.logger.error(
                f"Collector '{self.collector_type}' initialized without 'file_path' in config."
                )
            self.file_path = None
        else:
            try:
                # Validate the path immediately upon initialization
                self.file_path = validate_safe_path(raw_path, base_dir=Path.cwd())
            except PathValidationError as e:
                self.logger.error(f"Security violation: Invalid file path '{raw_path}': {e}")
                self.file_path = None

    async def fetch_raw_data(self, **kwargs) ->list[dict[str, Any]]:
        """
        Asynchronously reads data from a local file and returns it as a list of dictionaries.
        """
        if not self.file_path:
            return []
        self.logger.info(
            f'Fetching raw data from local {self.file_type} file: {self.file_path}'
            )
        try:
            if self.file_type == 'csv':
                df = await asyncio.to_thread(pd.read_csv, self.file_path)
            elif self.file_type == 'parquet':
                df = await asyncio.to_thread(pd.read_parquet, self.file_path)
            else:
                raise ValueError(f'Unsupported file type: {self.file_type}')
            if df.empty:
                self.logger.warning(f'File at {self.file_path} is empty.')
                return []
            if self.date_col and self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col]
                    ).dt.strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(
                f'Successfully loaded {len(df)} rows from {self.file_path}')
            return df.to_dict('records')
        except FileNotFoundError:
            self.logger.error(f'Local file not found: {self.file_path}')
            return []
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            raise RuntimeError(f"Failed to read local file {self.file_path}") from e

    async def post_process_new_records(self, records: pd.DataFrame
        ) ->pd.DataFrame:
        """
        Optional post-processing for local files.
        Ensures consistent naming if required by the target table.
        """
        return records
