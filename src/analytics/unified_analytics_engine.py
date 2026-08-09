"""
Unified Analytics Engine
Orchestrates the execution of various analytical modules in a parallelized and cached environment.
"""
import hashlib
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

    #: Seconds one analyzer may take before the engine gives up on it.
    DEFAULT_ANALYZER_TIMEOUT = 120

    def __init__(self, config_manager: UnifiedConfigManager):
        """
        Initializes the analytics engine.

        Args:
            config_manager: UnifiedConfigManager instance for loading engine and analyzer settings.
        """
        self.config_manager = config_manager
        self.analyzers: dict[str, IAnalyzer] = {}
        self.analyzer_data_map: dict[str, list[str]] = {}
        self.analyzer_registration_report: dict[str, dict[str, Any]] = {}
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
            analyzer_name = config.get('name', config.get('class', 'unknown').lower())
            if config.get('enabled', True) is not True:
                self.analyzer_registration_report[analyzer_name] = {
                    'status': 'disabled',
                    'reason': config.get('disabled_reason', 'disabled_in_config'),
                }
                continue
            try:
                module_path = config['module']
                class_name = config['class']
                params = config.get('params', {})
                module = importlib.import_module(module_path)
                analyzer_class = getattr(module, class_name)
                analyzer_instance = self._instantiate_analyzer(
                    analyzer_class,
                    params,
                )
                if isinstance(analyzer_instance, IAnalyzer):
                    self.register_analyzer(analyzer_instance, name=
                        analyzer_name)
                    self.analyzer_data_map[analyzer_name] = config.get(
                        'data_mapping', [])
                    self.analyzer_registration_report[analyzer_name] = {
                        'status': 'registered',
                        'required_inputs': config.get('data_mapping', []),
                        'class_path': f'{module_path}.{class_name}',
                    }
                else:
                    logger.warning(
                        f"Class '{class_name}' from '{module_path}' is not a valid IAnalyzer instance."
                        )
                    self.analyzer_registration_report[analyzer_name] = {
                        'status': 'rejected_not_analyzer',
                        'class_path': f'{module_path}.{class_name}',
                    }
            except (
                ImportError,
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                OSError,
            ) as e:
                logger.exception(
                    f"Failed to register analyzer '{config.get('name', 'unknown')}': {e}"
                    )
                self.analyzer_registration_report[analyzer_name] = {
                    'status': 'registration_failed',
                    'error': f'{type(e).__name__}: {e}',
                }

    def _instantiate_analyzer(
        self,
        analyzer_class: type,
        params: dict[str, Any],
    ) -> Any:
        try:
            return analyzer_class(**params)
        except TypeError as direct_error:
            try:
                return analyzer_class(config=params)
            except TypeError:
                raise direct_error

    def register_analyzer(self, analyzer: IAnalyzer, name: str):
        """Registers a single analyzer instance into the engine registry."""
        if not isinstance(analyzer, IAnalyzer):
            raise TypeError(
                f"Object '{name}' does not implement the IAnalyzer interface.")
        self.analyzers[name] = analyzer
        logger.info(f'Registered analyzer component: {name}')

    def _analysis_contract_payload(self) -> dict[str, Any]:
        """Return the analyzer-suite contract that makes cached results valid."""
        return {
            'configured_analyzers': self.analyzer_configs,
            'registered_analyzers': sorted(self.analyzers),
            'data_mapping': {
                name: list(keys)
                for name, keys in sorted(self.analyzer_data_map.items())
            },
            'registration_status': {
                name: details.get('status', 'unknown')
                for name, details in sorted(
                    self.analyzer_registration_report.items()
                )
            },
        }

    def _analysis_contract_hash(self) -> str:
        deterministic_json = json.dumps(
            self._analysis_contract_payload(),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(deterministic_json.encode()).hexdigest()

    def _generate_data_hash(self, data_map: dict[str, Any]) ->str:
        """
        Generates a stable fingerprint (MD5 hash) for input datasets to support result caching.
        """
        try:
            stable_repr: dict[str, Any] = {}
            stable_repr['_analysis_contract_hash'] = (
                self._analysis_contract_hash()
            )
            for key in sorted(data_map.keys()):
                value = data_map[key]
                if isinstance(value, pd.DataFrame):
                    content_hash = hashlib.sha256(
                        pd.util.hash_pandas_object(
                            value,
                            index=True,
                            categorize=True,
                        ).values.tobytes()
                    ).hexdigest()
                    stable_repr[key] = {
                        'shape': value.shape,
                        'columns': list(value.columns),
                        'dtypes': [str(dtype) for dtype in value.dtypes],
                        'content_hash': content_hash,
                    }
                elif isinstance(value, pd.Series):
                    content_hash = hashlib.sha256(
                        pd.util.hash_pandas_object(
                            value,
                            index=True,
                            categorize=True,
                        ).values.tobytes()
                    ).hexdigest()
                    stable_repr[key] = {
                        'shape': value.shape,
                        'name': value.name,
                        'dtype': str(value.dtype),
                        'content_hash': content_hash,
                    }
                else:
                    stable_repr[key] = str(value)
            deterministic_json = json.dumps(stable_repr, sort_keys=True)
            return hashlib.sha256(deterministic_json.encode()).hexdigest()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA
            logger.exception(f'Виникла помилка: {e}')
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'Fallback hashing triggered due to complex types: {e}')
            hash_input = ''
            for key, value in sorted(data_map.items()):
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    sample = value.head(3)
                    hash_input += (
                        f"{key}_{value.shape}_"
                        f"{hashlib.sha256(sample.to_json(date_format='iso', orient='split').encode()).hexdigest()}"
                        )
                else:
                    hash_input += f'{key}_{str(value)}'
            return hashlib.sha256(hash_input.encode()).hexdigest()

    def run_full_analysis(self, data_map: dict[str, Any], *,
        timeout: float | None = None, **kwargs) ->dict[str, Any]:
        """
        Executes all registered analyzers in parallel using the thread pool.

        Workflow:
        1. Generate input data fingerprint.
        2. Check cache for existing results.
        3. If no cache, route relevant data slices to analyzers.
        4. Collect and aggregate parallel results with safety timeouts.

        `timeout` is the engine's own budget per analyzer, declared here
        rather than read out of **kwargs. kwargs is forwarded verbatim to
        every analyzer's analyze(), so a timeout smuggled through it arrives
        as an argument each analyzer must tolerate -- and one that does not
        fails for a reason unrelated to analysis. Stage 7 passing timeout=30
        broke a caller this way on 2026-08-09.
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
        results: dict[str, Any] = {}
        routing_status: dict[str, dict[str, Any]] = {}
        for name, analyzer in self.analyzers.items():
            try:
                input_data = self._get_data_for_analyzer(name, data_map)
            except ConfigurationError as exc:
                required = self.analyzer_data_map.get(name, [])
                missing = [key for key in required if key not in data_map]
                results[name] = {
                    'status': 'skipped_missing_inputs',
                    'required_inputs': required,
                    'missing_inputs': missing,
                    'supporting_review_only': True,
                }
                routing_status[name] = {
                    'status': 'skipped_missing_inputs',
                    'missing_inputs': missing,
                }
                logger.info("Skipping analyzer '%s': %s", name, exc)
                continue
            if input_data is not None:
                futures[name] = self.thread_pool.submit(analyzer.analyze,
                    input_data, **kwargs)
                routing_status[name] = {'status': 'submitted'}
            else:
                logger.warning(
                    f"Skipping analyzer '{name}': dependent data keys not found in data_map."
                    )
                results[name] = {
                    'status': 'skipped_no_input_contract',
                    'supporting_review_only': True,
                }
                routing_status[name] = {'status': 'skipped_no_input_contract'}
        timeout = self.DEFAULT_ANALYZER_TIMEOUT if timeout is None else timeout
        for name, future in futures.items():
            try:
                raw_result = future.result(timeout=timeout)
                results[name] = self._normalize_analyzer_result(raw_result)
                routing_status[name] = {'status': 'executed'}
            except FuturesTimeoutError:
                # A timeout must SAY it timed out.
                #
                # concurrent.futures.TimeoutError carries no message, so
                # str(e) is the empty string -- and the failure was recorded
                # as {'error': '', 'status': 'failed'}. On the 2026-08-09 run
                # that is what 54 of 66 feature_drift results looked like:
                # failures with no stated reason, indistinguishable from a
                # crash, and reported upstream as "checked without errors".
                # Reconstructing that they were timeouts took reading the
                # artifacts. The number that explains them belongs in them.
                message = f'timed out after {timeout}s'
                logger.error(f"Analyzer '{name}' {message}")
                results[name] = {
                    'error': message,
                    'status': 'failed',
                    'timed_out': True,
                    'timeout_seconds': timeout,
                    'supporting_review_only': True,
                }
                routing_status[name] = {
                    'status': 'failed',
                    'error_type': 'TimeoutError',
                    'timeout_seconds': timeout,
                }
            except Exception as e:
                logger.exception(
                    f"Parallel execution failed for analyzer '{name}': {e}"
                    )
                results[name] = {
                    'error': str(e) or f'{type(e).__name__} (no message)',
                    'status': 'failed',
                    'supporting_review_only': True,
                }
                routing_status[name] = {
                    'status': 'failed',
                    'error_type': type(e).__name__,
                }
        results['_analysis_coverage'] = self._analysis_coverage(
            routing_status
        )
        self.results_manager.cache_analysis(data_hash, results)
        return results

    def _normalize_analyzer_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, pd.DataFrame):
            return {
                'status': 'completed',
                'output_type': 'dataframe_summary',
                'row_count': len(result),
                'columns': [str(column) for column in result.columns],
                'latest_record': (
                    result.tail(1).to_dict(orient='records')[0]
                    if not result.empty
                    else None
                ),
                'supporting_review_only': True,
            }
        if isinstance(result, pd.Series):
            return {
                'status': 'completed',
                'output_type': 'series_summary',
                'row_count': len(result),
                'name': str(result.name) if result.name is not None else None,
                'latest_value': result.iloc[-1] if not result.empty else None,
                'supporting_review_only': True,
            }
        if isinstance(result, dict):
            return {
                'status': result.get('status', 'completed'),
                **result,
                'supporting_review_only': True,
            }
        return {
            'status': 'completed',
            'output_type': type(result).__name__,
            'value': result,
            'supporting_review_only': True,
        }

    def _analysis_coverage(
        self,
        routing_status: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        statuses = {
            name: details.get('status', 'unknown')
            for name, details in routing_status.items()
        }
        registration_statuses = {
            name: details.get('status', 'unknown')
            for name, details in self.analyzer_registration_report.items()
        }
        return {
            'status': 'stage7_analyzer_coverage_recorded',
            'analysis_contract_hash': self._analysis_contract_hash(),
            'configured_count': len(self.analyzer_configs),
            'registered_count': len(self.analyzers),
            'executed': sorted(
                name for name, status in statuses.items()
                if status == 'executed'
            ),
            'skipped_missing_inputs': sorted(
                name for name, status in statuses.items()
                if status == 'skipped_missing_inputs'
            ),
            'failed': sorted(
                name for name, status in statuses.items()
                if status == 'failed'
            ),
            'disabled': sorted(
                name for name, status in registration_statuses.items()
                if status == 'disabled'
            ),
            'registration_failed': sorted(
                name for name, status in registration_statuses.items()
                if status in {'registration_failed', 'rejected_not_analyzer'}
            ),
            'routing': routing_status,
            'registration': self.analyzer_registration_report,
            'evidence_class': 'supporting_analysis_not_locked_evidence',
            'can_promote_model': False,
            'can_trade': False,
        }

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
            'orchestration_map': self.analyzer_data_map,
            'registration_report': self.analyzer_registration_report}

    def get_registered_components(self) ->dict[str, list[str]]:
        """Returns the list of registered analysis components."""
        return {'analyzers': list(self.analyzers.keys())}
