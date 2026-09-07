# src/data/collectors/base_collector.py

import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

import httpx

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    collector_type: str = "default"

    # Per-class fingerprint of the collector's own source, computed once.
    _cache_version_cache: dict[str, str] = {}

    @property
    def cache_version(self) -> str:
        """
        Fingerprint of this collector's source, for use as a cache key version.

        A cached *payload* — a records list, a fetched frame — has the shape the
        collector gave it. Change the collector's parsing or normalisation and
        last week's payload is a different shape than this week's code expects,
        yet the old key still matches and the old payload is served for the rest
        of its TTL. Folding the source fingerprint into the key makes a
        collector invalidate its own entries when it changes, and only its own.

        Deliberately NOT applied to per-record "already seen this hash" markers:
        their meaning is content identity, which no code change alters, and
        versioning them would re-ingest the entire history on every edit.
        """
        cls = type(self)
        cached = BaseCollector._cache_version_cache.get(cls.__qualname__)
        if cached is not None:
            return cached
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            # Source unavailable (zipped install, dynamically built class).
            # Fall back to the class name: no invalidation on change, but no
            # crash either, and never a false match against another collector.
            source = cls.__qualname__
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        BaseCollector._cache_version_cache[cls.__qualname__] = digest
        return digest

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        self.collector_type = configs.get('type', self.collector_type)
        self.logger = ProjectLogger.get_logger(f"{self.collector_type}_collector")
        self.configs = configs
        self.http_client_factory = http_client_factory
        self.db_manager = db_manager
        self.cache_manager = cache_manager

    def get_client(self, **kwargs) -> httpx.AsyncClient:
        """Helper to get a configured client from the factory."""
        return self.http_client_factory.get_http_client(**kwargs)

    @abstractmethod
    async def run(self, tickers: list[str], **kwargs) -> Any | None:
        """Main method to execute the data collection logic."""
        pass
