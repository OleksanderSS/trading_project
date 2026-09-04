import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.core.logging.logger import ProjectLogger
from src.core.security.path_validator import PathValidationError, validate_safe_path
from src.utils.path_safety import get_path_safety

logger = ProjectLogger.get_logger('FileManager')


class FileManager:
    """Provides a centralized and robust interface for file operations with atomic writes and background tasks."""

    def __init__(self, base_dir: (str | Path | None)=None, max_workers: int=4):
        path_safety = get_path_safety()
        self.base_dir = Path(base_dir).resolve() if base_dir else path_safety.get_project_root()
        self.logger = logger
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _resolve_path(self, file_path: (str | Path)) ->Path:
        """Resolves a given path to be absolute and validated against the base directory."""
        try:
            return validate_safe_path(file_path, base_dir=self.base_dir)
        except PathValidationError as e:
            self.logger.error(f"Security violation attempting to access path '{file_path}': {e}")
            raise

    def ensure_directory(self, dir_path: (str | Path)) ->Path:
        """Ensures that a directory exists, creating it if necessary."""
        path = self._resolve_path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def find_files(self, pattern: str, search_dir: (str | Path | None)=None
        ) ->list[Path]:
        """Finds files matching a glob pattern within a specified directory."""
        search_path = self._resolve_path(search_dir
            ) if search_dir else self.base_dir
        return list(search_path.glob(pattern))

    def _atomic_write(self, file_path: Path, write_func: Callable[[Path],
        None], validate_func: (Callable[[Path], bool] | None)=None):
        """Performs an atomic write using a temporary file and optional validation."""
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            write_func(temp_path)
            if validate_func and not validate_func(temp_path):
                raise OSError(
                    f'Integrity check failed for temporary file: {temp_path}')
            os.replace(temp_path, file_path)
            self.logger.info(f'Successfully saved data to {file_path}')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError, OSError) as e:
            if temp_path.exists():
                temp_path.unlink()
            self.logger.error(f'Failed to save data to {file_path}: {e}',
                exc_info=True)
            raise

    def save_yaml(self, data: dict[str, Any], file_path: (str | Path),
        async_save: bool=False) ->None:
        """Saves a dictionary to a YAML file atomically."""
        path = self._resolve_path(file_path)
        self.ensure_directory(path.parent)

        def write_task(p: Path):
            with open(p, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)

        def validate_task(p: Path) ->bool:
            try:
                with open(p, encoding='utf-8') as f:
                    yaml.safe_load(f)
                return True
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                return False
        if async_save:
            self._executor.submit(self._atomic_write, path, write_task,
                validate_task)
        else:
            self._atomic_write(path, write_task, validate_task)

    def load_yaml(self, file_path: (str | Path)) ->(dict[str, Any] | None):
        """Loads a dictionary from a YAML file."""
        path = self._resolve_path(file_path)
        if not path.exists():
            self.logger.warning(f'File not found: {path}')
            return None
        try:
            with open(path, encoding='utf-8') as f:
                data: dict[str, Any] | None = yaml.safe_load(f)
        except (yaml.YAMLError, OSError, ValueError, TypeError) as e:
            # yaml.YAMLError inherits from Exception, NOT from ValueError, so
            # the project's usual five-tuple missed it and a malformed YAML
            # escaped as a raw ParserError while malformed JSON -- whose
            # JSONDecodeError IS a ValueError -- came back as the intended
            # RuntimeError. Two loaders in the same class behaving differently
            # on the same class of failure.
            self.logger.error(f'Failed to load YAML from {path}: {e}',
                exc_info=True)
            raise RuntimeError(f"Failed to load YAML from {path}") from e

        if data is None:
            self.logger.warning(f'YAML file is empty: {path}')
            return None
        if not isinstance(data, dict):
            # Previously returned None here in silence, which the caller
            # cannot tell apart from "file not found". A config file that
            # exists but is shaped wrong was skipped without explanation.
            self.logger.warning(
                f'YAML at {path} parsed as {type(data).__name__}, not a '
                f'mapping; ignoring its contents.'
            )
            return None

        self.logger.info(f'Loaded YAML from {path}')
        return data

    def save_json(self, data: dict[str, Any], file_path: (str | Path),
        async_save: bool=False) ->None:
        """Saves a dictionary to a JSON file atomically."""
        path = self._resolve_path(file_path)
        self.ensure_directory(path.parent)

        def write_task(p: Path):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        def validate_task(p: Path) ->bool:
            try:
                with open(p, encoding='utf-8') as f:
                    json.load(f)
                return True
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                return False
        if async_save:
            self._executor.submit(self._atomic_write, path, write_task,
                validate_task)
        else:
            self._atomic_write(path, write_task, validate_task)

    def load_json(self, file_path: (str | Path)) ->(dict[str, Any] | None):
        """Loads a dictionary from a JSON file."""
        path = self._resolve_path(file_path)
        if not path.exists():
            self.logger.warning(f'File not found: {path}')
            return None
        try:
            with open(path, encoding='utf-8') as f:
                data: dict[str, Any] | Any = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            # OSError added for symmetry with load_yaml: a permission or disk
            # error should raise the same contextual RuntimeError as a parse
            # failure, not a bare OSError from three frames down.
            self.logger.error(f'Failed to load JSON from {path}: {e}',
                exc_info=True)
            raise RuntimeError(f"Failed to load JSON from {path}") from e

        if not isinstance(data, dict):
            self.logger.warning(
                f'JSON at {path} parsed as {type(data).__name__}, not an '
                f'object; ignoring its contents.'
            )
            return None

        self.logger.info(f'Loaded JSON from {path}')
        return data

    def _remove_timezone(self, df: pd.DataFrame) ->pd.DataFrame:
        """Timezone stripped from every datetime column, WITHOUT touching `df`.

        This used to assign into its argument and return it, so the caller had
        to defend itself with a full deep copy of the frame -- and did, on
        every save, in both branches. `assign` builds the replacement columns
        and returns a new frame, so the caller needs no copy at all and a
        function named "remove" no longer edits what it was handed.

        Saving a frame does not modify it, so the other branch's copy was
        pointless outright.
        """
        tz_columns = [
            column for column in df.columns
            if pd.api.types.is_datetime64tz_dtype(df[column])
        ]
        if not tz_columns:
            return df
        return df.assign(**{
            column: df[column].dt.tz_localize(None) for column in tz_columns
        })

    def _write_dataframe(self, df: pd.DataFrame, path: Path, format: str, **kwargs) -> None:
        """Write DataFrame to file in specified format."""
        if format == 'parquet':
            df.to_parquet(path, **kwargs)
        elif format == 'csv':
            df.to_csv(path, index=False, **kwargs)
        elif format == 'json':
            df.to_json(path, orient='records', date_format='iso', **kwargs)
        else:
            raise ValueError(f'Unsupported format: {format}')

    def _validate_dataframe(self, path: Path, format: str, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame was written correctly."""
        try:
            if format == 'parquet':
                pd.read_parquet(path, columns=[df.columns[0]])
            elif format == 'csv':
                pd.read_csv(path, nrows=1)
            elif format == 'json':
                pd.read_json(path, nrows=1)
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return False

    def save_dataframe(self, df: pd.DataFrame, file_path: (str | Path),
        format: str='parquet', remove_tz: bool=False, async_save: bool=
        False, **kwargs) ->None:
        """
        Saves a DataFrame to a file atomically in the specified format.

        CodeScene/Ruff: Complex Method acceptable - DataFrame saving requires multiple
        conditional branches to handle different file formats (parquet, csv, json, pickle),
        timezone handling, atomic writes, and error recovery. This complexity is inherent
        to flexible data persistence.
        """
        path = self._resolve_path(file_path)
        self.ensure_directory(path.parent)
        df_to_save = self._remove_timezone(df) if remove_tz else df

        def write_task(p: Path):
            self._write_dataframe(df_to_save, p, format, **kwargs)

        def validate_task(p: Path) ->bool:
            return self._validate_dataframe(p, format, df_to_save)

        if async_save:
            self._executor.submit(self._atomic_write, path, write_task,
                validate_task)
        else:
            self._atomic_write(path, write_task, validate_task)

    def load_dataframe(self, file_path: (str | Path), format: str='parquet',
        **kwargs) ->(pd.DataFrame | None):
        """
        Loads a DataFrame from a file.
        """
        path = self._resolve_path(file_path)
        if not path.exists():
            self.logger.warning(f'File not found: {path}')
            return None
        try:
            if format == 'parquet':
                df = pd.read_parquet(path, **kwargs)
            elif format == 'csv':
                df = pd.read_csv(path, **kwargs)
            elif format == 'json':
                df = pd.read_json(path, **kwargs)
            else:
                raise ValueError(f'Unsupported format: {format}')
            self.logger.info(f'Loaded {len(df)} rows from {path}')
            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Failed to load DataFrame from {path}: {e}',
                exc_info=True)
            raise RuntimeError(f"Failed to load DataFrame from {path}") from e

    def cleanup_old_files(self, directory: (str | Path), max_age_days: int=
        7, pattern: str='*') ->None:
        """
        Removes files older than max_age_days from the specified directory.
        """
        path = self._resolve_path(directory)
        if not path.exists() or not path.is_dir():
            self.logger.warning(
                f'Cleanup skipped: {path} is not a valid directory.')
            return
        now = time.time()
        cutoff = now - max_age_days * 86400
        deleted_count = 0
        for f in path.glob(pattern):
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    deleted_count += 1
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.logger.warning(f'Failed to delete {f}: {e}')
                    raise
        if deleted_count > 0:
            self.logger.info(
                f'Cleanup in {path}: Deleted {deleted_count} files older than {max_age_days} days.'
                )
