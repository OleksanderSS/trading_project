"""
Unified Analytics Engine
Orchestrates the execution of various analytical modules in a parallelized and cached environment.
"""
import hashlib
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.analytics.interfaces import IAnalyzer
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class UnifiedAnalyticsEngine:
    """
    Main analytical engine responsible for orchestrating multiple analysis modules.

    Key Responsibilities:
    - Automatically loads and registers analyzers (IAnalyzer) based on centralized configuration.
    - Executes analysis tasks in parallel using a thread pool for high throughput.
    - Manages data routing to each analyzer according to 'data_mapping' definitions.
    - Implements a caching mechanism to prevent redundant computations for identical input datasets.
    - Persists and manages analytical results via ModelResultsManager.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """
        Initializes the analytics engine.

        Args:
            config_manager: UnifiedConfigManager instance for loading engine and analyzer settings.
        """
        self.config_manager = config_manager
        self.analyzers: dict[str, IAnalyzer] = {}
        self.analyzer_data_map: dict[str, list[str]] = {}
        engine_config = self.config_manager.get('analysis.engine', {})
        self.max_workers = engine_config.get('max_workers', 4)
        self.analyzer_configs = engine_config.get('analyzers', [])
        self.results_manager = ModelResultsManager()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._register_analyzers_from_config()
        logger.info(
            f'UnifiedAnalyticsEngine initialized with {len(self.analyzers)} analyzers.'
            )

    def _register_analyzers_from_config(self):
        """Dynamically imports and initializes analyzer instances based on configuration."""
        for config in self.analyzer_configs:
            try:
                module_path = config['module']
                class_name = config['class']
                analyzer_name = config.get('name', class_name.lower())
                params = config.get('params', {})
                module = importlib.import_module(module_path)
                analyzer_class = getattr(module, class_name)
                analyzer_instance = analyzer_class(**params)
                if isinstance(analyzer_instance, IAnalyzer):
                    self.register_analyzer(analyzer_instance, name=
                        analyzer_name)
                    self.analyzer_data_map[analyzer_name] = config.get(
                        'data_mapping', [])
                else:
                    logger.warning(
                        f"Class '{class_name}' from '{module_path}' is not a valid IAnalyzer instance."
                        )
            except (ImportError, AttributeError, KeyError, TypeError) as e:
                logger.error(
                    f"Failed to register analyzer '{config.get('name', 'unknown')}': {e}"
                    , exc_info=True)

    def register_analyzer(self, analyzer: IAnalyzer, name: str):
        """Registers a single analyzer instance into the engine registry."""
        if not isinstance(analyzer, IAnalyzer):
            raise TypeError(
                f"Object '{name}' does not implement the IAnalyzer interface.")
        self.analyzers[name] = analyzer
        logger.info(f'Registered analyzer component: {name}')

    def _generate_data_hash(self, data_map: dict[str, Any]) ->str:
        """
        Generates a stable fingerprint (MD5 hash) for input datasets to support result caching.
        """
        try:
            stable_repr: dict[str, Any] = {}
            for key in sorted(data_map.keys()):
                value = data_map[key]
                if isinstance(value, pd.DataFrame):
                    sample = value.head(10).tail(5)
                    stable_repr[key] = {'shape': value.shape, 'columns':
                        list(value.columns), 'sample_hash': hashlib.sha256(
                        sample.to_json(date_format='iso', orient='split').
                        encode()).hexdigest()}
                elif isinstance(value, pd.Series):
                    sample = value.head(10)
                    stable_repr[key] = {'shape': value.shape, 'name': value
                        .name, 'sample_hash': hashlib.sha256(sample.to_json
                        (date_format='iso', orient='split').encode()).
                        hexdigest()}
                else:
                    stable_repr[key] = str(value)
            deterministic_json = json.dumps(stable_repr, sort_keys=True)
            return hashlib.sha256(deterministic_json.encode()).hexdigest()
        except Exception as e:  # audit-ignore: EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'Fallback hashing triggered due to complex types: {e}')
            hash_input = ''
            for key, value in sorted(data_map.items()):
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    sample = value.head(3)
                    hash_input += (
                        f"{key}_{value.shape}_{hash(sample.to_json(date_format='iso', orient='split'))}"
                        )
                else:
                    hash_input += f'{key}_{str(value)}'
            return hashlib.sha256(hash_input.encode()).hexdigest()

    def run_full_analysis(self, data_map: dict[str, Any], **kwargs) ->dict[
        str, Any]:
        """
        Executes all registered analyzers in parallel using the thread pool.

        Workflow:
        1. Generate input data fingerprint.
        2. Check cache for existing results.
        3. If no cache, route relevant data slices to analyzers.
        4. Collect and aggregate parallel results with safety timeouts.
        """
        data_hash = self._generate_data_hash(data_map)
        cached_results = self.results_manager.get_cached_analysis(data_hash)
        if cached_results:
            logger.info('Retrieved analysis results from persistent cache.')
            return cached_results
        logger.info(
            f'Commencing parallel analysis suite with {len(self.analyzers)} modules.'
            )
        futures = {}
        for name, analyzer in self.analyzers.items():
            input_data = self._get_data_for_analyzer(name, data_map)
            if input_data is not None:
                futures[name] = self.thread_pool.submit(analyzer.analyze,
                    input_data, **kwargs)
            else:
                logger.warning(
                    f"Skipping analyzer '{name}': dependent data keys not found in data_map."
                    )
        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=120)
            except Exception as e:
                logger.error(
                    f"Parallel execution failed for analyzer '{name}': {e}",
                    exc_info=True)
                results[name] = {'error': str(e), 'status': 'failed'}
        self.results_manager.cache_analysis(data_hash, results)
        return results

    def _get_data_for_analyzer(self, analyzer_name: str, data_map: dict[str,
        Any]) ->Any | None:
        """Routes specific data subsets to an analyzer based on its configured requirements."""
        required_keys = self.analyzer_data_map.get(analyzer_name, [])
        if not required_keys:
            logger.warning(
                f"No data_mapping defined for analyzer '{analyzer_name}'. Skipping."
                )
            return None
        if not all(key in data_map for key in required_keys):
            missing = [k for k in required_keys if k not in data_map]
            error_msg = f"Insufficient data for '{analyzer_name}'. Missing keys: {missing}."
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        if len(required_keys) == 1:
            return data_map[required_keys[0]]
        return {key: data_map[key] for key in required_keys}

    def get_contextual_report(self) ->dict[str, Any]:
        """Generates an observability report on the engine's current operational state."""
        return {'engine_status': 'operational', 'active_analyzers_count':
            len(self.analyzers), 'registered_modules': list(self.analyzers.
            keys()), 'max_concurrency': self.max_workers,
            'orchestration_map': self.analyzer_data_map}

    def get_registered_components(self) ->dict[str, list[str]]:
        """Returns the list of registered analysis components."""
        return {'analyzers': list(self.analyzers.keys())}
