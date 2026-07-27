
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

class QueryCache:
    """
    A simple, file-based cache manager for storing the results of data queries (DataFrames)
    in the efficient Parquet format.
    """

    def __init__(self, cache_dir: str = "cache/query_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key_from_query(self, query: str) -> str:
        """Generates a unique and filesystem-safe key from an SQL query."""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str, max_age_hours: int = 24) -> pd.DataFrame | None:
        """
        Retrieves a DataFrame from the cache if it exists and is not expired.

        Args:
            query (str): The SQL query that was used to generate the data.
            max_age_hours (int): The maximum age of the cache file in hours.

        Returns:
            Optional[pd.DataFrame]: The cached DataFrame, or None if not found or expired.
        """
        cache_key = self._generate_key_from_query(query)
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if not cache_file.exists():
            return None

        # Check cache file age
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_mod_time > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for query hash: {cache_key}. Cleaning up.")
            cache_file.unlink()
            return None

        try:
            df = pd.read_parquet(cache_file)
            logger.debug(f"Loaded DataFrame from cache for query hash: {cache_key}")
            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}. Removing corrupt file.")
            cache_file.unlink()
            return None

    def set(self, query: str, data: pd.DataFrame):
        """
        Saves a DataFrame to the cache.

        Args:
            query (str): The SQL query used to generate the data.
            data (pd.DataFrame): The DataFrame to be cached.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            return

        cache_key = self._generate_key_from_query(query)
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        try:
            data.to_parquet(cache_file, compression='snappy', index=False)
            logger.debug(f"Saved DataFrame to cache for query hash: {cache_key}")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to save data to cache file {cache_file}: {e}", exc_info=True)

    def clear(self):
        """Clears the entire query cache."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
        logger.info("Cleared all query cache.")

