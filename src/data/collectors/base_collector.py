# src/data/collectors/base_collector.py

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
