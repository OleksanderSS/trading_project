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
        if self.db_manager:
            try:
                self.cache_manager = CacheManager(data_manager=self.
                    db_manager, config_manager=self.config_manager)
                self.logger.info('CacheManager initialized successfully.')
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(
                    f'CacheManager initialization failed: {e}. Continuing without cache.'
                    )
                self.cache_manager = None
                raise
        else:
            self.cache_manager = None
            self.logger.warning(
                'CacheManager not initialized: db_manager is unavailable.')
        self._collector_classes = self._discover_collector_classes()

    def _discover_collector_classes(self) ->dict[str, type[BaseCollector]]:
        """Динамічно знаходить всі класи колекторів."""
        class_map = {}
        package_path = os.path.dirname(__file__)
        package_name = 'src.data.collectors'
        for _, module_name, _ in pkgutil.walk_packages([package_path],
            prefix=f'{package_name}.'):
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseCollector
                        ) and obj is not BaseCollector:
                        collector_type = getattr(obj, 'collector_type', None)
                        if collector_type and collector_type != 'default':
                            if collector_type in class_map:
                                self.logger.warning(
                                    f"Duplicate collector_type '{collector_type}'. Overwriting with {name}."
                                    )
                            class_map[collector_type] = obj
            except Exception as e:
                self.logger.error(f'Failed to import {module_name}: {e}')
        self.logger.info(
            f'Discovered {len(class_map)} collector classes: {list(class_map.keys())}'
            )
        return class_map

    def get_collector(self, name: str) ->BaseCollector | None:
        """Створює екземпляр одного колектора за назвою конфігурації."""
        collector_params = self.collectors_config.get(name)
        if not collector_params or not collector_params.get('enabled', False):
            return None
        collector_type = collector_params.get('type')
        CollectorClass = self._collector_classes.get(collector_type)
        if not CollectorClass:
            self.logger.error(
                f"No collector class found for type '{collector_type}'. Available: {list(self._collector_classes.keys())}"
                )
            return None
        try:
            return CollectorClass(configs=collector_params,
                http_client_factory=self.http_client_factory,
                config_manager=self.config_manager, db_manager=self.
                db_manager, cache_manager=self.cache_manager)
        except Exception as e:
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
