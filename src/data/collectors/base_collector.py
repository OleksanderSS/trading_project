# src/data/collectors/base_collector.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.core.cache.cache_manager import CacheManager
from src.data.management.data_manager import DataManager # Додаємо імпорт

class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    collector_type: str = "default"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, cache_manager: Optional[CacheManager] = None, **kwargs):
        self.collector_type = configs.get('type', self.collector_type)
        self.logger = ProjectLogger.get_logger(f"{self.collector_type}_collector")
        self.configs = configs
        self.http_client_factory = http_client_factory
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.rate_limiter = kwargs.get('rate_limiter')

    @abstractmethod
    async def run(self, tickers: List[str], **kwargs) -> Optional[Any]:
        """Main method to execute the data collection logic."""
        pass