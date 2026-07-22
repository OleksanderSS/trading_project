import importlib
import inspect
import os
import pkgutil
from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class CollectorFactory:
    """Dynamically finds and creates collector instances."""

    def __init__(self, configs: dict | None=None, http_client_factory:
        HttpClientFactory | None=None, config_manager: UnifiedConfigManager | None=None, db_manager: DataManager | None=None,
        cache_manager: Any | None=None):
        self.logger = ProjectLogger.get_logger('CollectorFactory')
        self.collectors_config = configs or {}
        self.http_client_factory = http_client_factory or HttpClientFactory()
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.cache_manager = self._initialize_cache_manager()
        self._collector_classes = self._discover_collector_classes()

    def _initialize_cache_manager(self):
        """Initialize cache manager if db_manager is available."""
        if self.db_manager:
            return self._create_cache_manager()
        else:
            self.logger.warning(
                'CacheManager not initialized: db_manager is unavailable.')
            return None

    def _create_cache_manager(self):
        """Create cache manager with error handling."""
        try:
            cache_manager = CacheManager(data_manager=self.db_manager, config_manager=self.config_manager)
            self.logger.info('CacheManager initialized successfully.')
            return cache_manager
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'CacheManager initialization failed: {e}. Continuing without cache.')
            raise

    def _discover_collector_classes(self) ->dict[str, type[BaseCollector]]:
        """Динамічно знаходить всі класи колекторів."""
        class_map = {}
        package_path = os.path.dirname(__file__)
        package_name = 'src.data.collectors'

        for _, module_name, _ in pkgutil.walk_packages([package_path], prefix=f'{package_name}.'):
            self._process_module_for_collectors(module_name, class_map)

        self.logger.info(
            f'Discovered {len(class_map)} collector classes: {list(class_map.keys())}'
            )
        return class_map

    def _process_module_for_collectors(self, module_name: str, class_map: dict):
        """Process a single module to find collector classes."""
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                self._register_collector_if_valid(obj, name, class_map)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Failed to import {module_name}: {e}')

    def _register_collector_if_valid(self, obj: type, name: str, class_map: dict):
        """Register a collector class if it's valid."""
        if issubclass(obj, BaseCollector) and obj is not BaseCollector:
            collector_type = getattr(obj, 'collector_type', None)
            if collector_type and collector_type != 'default':
                self._handle_duplicate_collector(collector_type, name, class_map)
                class_map[collector_type] = obj

    def _handle_duplicate_collector(self, collector_type: str, name: str, class_map: dict):
        """Handle duplicate collector types with warning."""
        if collector_type in class_map:
            self.logger.warning(
                f"Duplicate collector_type '{collector_type}'. Overwriting with {name}."
                )

    def get_collector(self, name: str) ->BaseCollector | None:
        """Створює екземпляр одного колектора за назвою конфігурації."""
        collector_params = self.collectors_config.get(name)
        if not self._is_collector_enabled(collector_params):
            return None

        collector_type = collector_params.get('type')
        CollectorClass = self._get_collector_class(collector_type)
        if not CollectorClass:
            return None

        return self._instantiate_collector(name, collector_type, CollectorClass, collector_params)

    def _is_collector_enabled(self, collector_params: dict | None) -> bool:
        """Check if collector is enabled in configuration."""
        return collector_params is not None and collector_params.get('enabled', False)

    def _get_collector_class(self, collector_type: str) -> type[BaseCollector] | None:
        """Get collector class by type."""
        CollectorClass = self._collector_classes.get(collector_type)
        if not CollectorClass:
            self.logger.error(
                f"No collector class found for type '{collector_type}'. Available: {list(self._collector_classes.keys())}"
                )
        return CollectorClass

    def _instantiate_collector(self, name: str, collector_type: str, CollectorClass: type, collector_params: dict) -> BaseCollector:
        """Instantiate collector with error handling."""
        try:
            return CollectorClass(configs=collector_params,
                http_client_factory=self.http_client_factory,
                config_manager=self.config_manager, db_manager=self.db_manager,
                cache_manager=self.cache_manager)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to instantiate '{name}' ({collector_type}): {e}", exc_info=True)
            raise RuntimeError(f"Failed to instantiate collector '{name}' ({collector_type})") from e

    def get_all_collectors(self) ->list[BaseCollector]:
        """Створює всі увімкнені колектори."""
        collectors = []
        for name in self.collectors_config.keys():
            collector = self.get_collector(name)
            if collector:
                collectors.append(collector)
        self.logger.info(f'Instantiated {len(collectors)} enabled collectors.')
        return collectors
