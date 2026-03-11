
# src/data/collectors/collector_factory.py

import inspect
import pkgutil
import importlib
import os
from typing import List, Dict, Type, Optional

from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.core.cache.cache_manager import CacheManager
from .base_collector import BaseCollector
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager

class CollectorFactory:
    """A factory class for discovering and creating collector instances."""

    def __init__(self, configs: Optional[Dict] = None, http_client_factory: Optional[HttpClientFactory] = None, config_manager: Optional[UnifiedConfigManager] = None, db_manager: Optional[DataManager] = None):
        self.logger = ProjectLogger.get_logger("CollectorFactory")
        self.collectors_config = configs or {}
        self.http_client_factory = http_client_factory or HttpClientFactory()
        self.config_manager = config_manager
        self.db_manager = db_manager
        
        # Instantiate CacheManager if a db_manager is available
        self.cache_manager = CacheManager(data_manager=self.db_manager) if self.db_manager else None
        if self.cache_manager:
            self.logger.info("CacheManager initialized successfully.")
        else:
            self.logger.warning("CacheManager could not be initialized because db_manager is not available.")

        self._collector_classes = self._discover_collector_classes()

    def _discover_collector_classes(self) -> Dict[str, Type[BaseCollector]]:
        """Dynamically discovers all collector classes."""
        class_map = {}
        package_path = os.path.dirname(__file__)
        package_name = 'src.data.collectors'

        for _, module_name, _ in pkgutil.walk_packages([package_path], prefix=f"{package_name}."):
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseCollector) and obj is not BaseCollector:
                        collector_type = getattr(obj, 'collector_type', None)
                        if collector_type:
                            if collector_type in class_map:
                                self.logger.warning(f"Duplicate collector_type '{collector_type}' found. Overwriting.")
                            class_map[collector_type] = obj
            except Exception as e:
                self.logger.error(f"Failed to discover or import from {module_name}: {e}")
        
        self.logger.info(f"Discovered {len(class_map)} collector classes: {list(class_map.keys())}")
        return class_map

    def get_collector(self, name: str) -> Optional[BaseCollector]:
        """Instantiates a single collector by its configuration name."""
        collector_params = self.collectors_config.get(name)
        if not collector_params or not collector_params.get("enabled", False):
            return None

        collector_type = collector_params.get("type")
        CollectorClass = self._collector_classes.get(collector_type)

        if not CollectorClass:
            self.logger.error(f"No collector class found for type '{collector_type}'.")
            return None

        try:
            # Pass the cache_manager and other dependencies to the collector constructor
            return CollectorClass(
                configs=collector_params, 
                http_client_factory=self.http_client_factory, 
                config_manager=self.config_manager, 
                db_manager=self.db_manager,
                cache_manager=self.cache_manager  # Pass the cache manager
            )
        except Exception as e:
            self.logger.error(f"Failed to instantiate collector '{name}': {e}", exc_info=True)
            return None

    def get_all_collectors(self) -> List[BaseCollector]:
        """Instantiates all enabled collectors."""
        collectors = []
        for name in self.collectors_config.keys():
            collector = self.get_collector(name)
            if collector:
                collectors.append(collector)
        
        self.logger.info(f"Successfully instantiated {len(collectors)} enabled collectors.")
        return collectors
