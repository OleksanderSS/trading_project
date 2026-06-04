
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class ObjectCache:
    """
    A simple, file-based cache manager for storing arbitrary Python objects
    using pickle serialization.

    This is suitable for caching objects like trained models, lists, dictionaries, etc.
    """

    def __init__(self, cache_dir: str = "cache/object_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generates a filesystem-safe path for a given key."""
        # Simple sanitization for the key to be used as a filename
        safe_key = "".join(c for c in key if c.isalnum() or c in ('_', '-'))
        return self.cache_dir / f"{safe_key}.pkl"

    def get(self, key: str, max_age_hours: int = 24*7) -> Any | None:
        """
        Retrieves an object from the cache if it exists and is not expired.

        Args:
            key (str): The unique key for the object.
            max_age_hours (int): The maximum age of the cache file in hours. Defaults to 7 days.

        Returns:
            Optional[Any]: The cached object, or None if not found or expired.
        """
        cache_file = self._get_cache_path(key)

        if not cache_file.exists():
            return None

        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_mod_time > timedelta(hours=max_age_hours):
            logger.info(f"Object cache for key '{key}' has expired. Cleaning up.")
            cache_file.unlink()
            return None

        try:
            with open(cache_file, "rb") as f:
                obj = pickle.load(f)
            logger.debug(f"Loaded object from cache for key: '{key}'")
            return obj
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Failed to unpickle cache file {cache_file}: {e}. Removing corrupt file.")
            cache_file.unlink()
            return None

    def set(self, key: str, obj: Any):
        """
        Saves an object to the cache.

        Args:
            key (str): The unique key for the object.
            obj (Any): The Python object to be cached.
        """
        cache_file = self._get_cache_path(key)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(obj, f)
            logger.debug(f"Saved object to cache for key: '{key}'")
        except Exception as e:
            logger.error(f"Failed to save object to cache file {cache_file}: {e}", exc_info=True)

    def clear(self, key: str | None = None):
        """
        Clears the cache.

        If a key is provided, only that entry is removed.
        If no key is provided, the entire object cache is cleared.
        """
        if key:
            cache_file = self._get_cache_path(key)
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared object cache for key: '{key}'")
        else:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            logger.info("Cleared all object cache.")
