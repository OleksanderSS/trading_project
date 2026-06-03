from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("LocalFileDataSource")


class LocalFileDataSource:
    """Config-driven adapter for loading local CSV or Parquet files."""

    def __init__(
        self,
        file_path: str | Path,
        file_type: str = "csv",
        date_col: str | None = None,
        **_: Any,
    ):
        self.file_path = Path(file_path)
        self.file_type = file_type.lower()
        self.date_col = date_col

    def load(self, **overrides: Any) -> pd.DataFrame:
        """Load the configured file as a DataFrame."""
        file_path = Path(overrides.get("file_path", self.file_path))
        file_type = overrides.get("file_type", self.file_type).lower()
        date_col = overrides.get("date_col", self.date_col)

        if not file_path.exists():
            raise FileNotFoundError(f"Local data source not found: {file_path}")

        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported local data source type: {file_type}")

        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        logger.info(f"Loaded local data source {file_path}: {df.shape}")
        return df

    def read(self, **overrides: Any) -> pd.DataFrame:
        """Alias for loader APIs that expect read()."""
        return self.load(**overrides)

    def fetch(self, **overrides: Any) -> pd.DataFrame:
        """Alias for loader APIs that expect fetch()."""
        return self.load(**overrides)
