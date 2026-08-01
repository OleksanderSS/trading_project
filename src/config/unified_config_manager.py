from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger
from src.core.security.secure_secrets_manager import SecretsManager

if TYPE_CHECKING:
    pass

logger = ProjectLogger.get_logger("UnifiedConfigManager")


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class DynamicConfig:
    """Wrapper for recursive attribute-selection from dictionary-based configuration."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DynamicConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the internal dictionary."""
        return self._data.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """Exports content back to a standard dictionary."""
        return self._data

    def __getattr__(self, name: str) -> Any:
        if name not in self._data:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._get_attribute_value(name)

    def _get_attribute_value(self, name: str) -> Any:
        """Get attribute value from internal data."""
        value = self._data[name]
        return DynamicConfig(value) if isinstance(value, dict) else value

    def __repr__(self) -> str:
        return str(self._data)


def _deep_merge(source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges source dictionary into destination."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination


class UnifiedConfigManager:
    """Manages system-wide configuration loading, validation, and secret synchronization."""

    # Class-level flag to prevent circular dependency during initialization
    _initializing = False

    def __init__(self, env: Environment = Environment.DEVELOPMENT, config_dir: str | Path | None = None,
                 create_paths: bool = True, resolve_secrets: bool = True, validate_cloud: bool = True):
        """
        Initializes the manager.

        Args:
            env: Active deployment environment.
            config_dir: Directory containing YAML configuration templates.
            create_paths: Whether to create directories specified in config (default: True).
                          Set to False for tests/devtools to avoid filesystem side effects.
            resolve_secrets: Whether to resolve secret placeholders (default: True).
                            Set to False for tests/devtools to avoid secret manager dependency.
            validate_cloud: Whether to validate cloud storage configuration (default: True).
                           Set to False for tests/devtools to avoid cloud validation.
        """
        self.env = env
        if config_dir:
            self.config_dir = Path(config_dir).resolve()
        else:
            # Default to the directory of this source file
            self.config_dir = Path(__file__).parent

        # Determine project root based on directory structure
        self.project_root = self.config_dir.parent.parent
        self.file_manager = FileManager(base_dir=self.project_root)
        self.merged_config: dict[str, Any] = {}
        self.feature_sets: dict[str, list[str]] = {}

        # Store side effect flags for reload()
        self._create_paths = create_paths
        self._resolve_secrets_flag = resolve_secrets
        self._validate_cloud_flag = validate_cloud

        UnifiedConfigManager._initializing = True
        try:
            self._load_and_resolve_configs()
            self._setup_dynamic_attributes()

            if validate_cloud:
                self.validate_configuration()

            if create_paths:
                self._ensure_paths_exist()

            # Resolve secrets after full initialization to avoid circular dependency
            if resolve_secrets:
                self._resolve_secrets_in_config()

            self.feature_sets = self._generate_feature_lists()
        finally:
            UnifiedConfigManager._initializing = False

        logger.info(f"UnifiedConfigManager initialized for '{self.env.value}' environment.")

    def reload(self):
        """Refreshes the configuration from disk without process restart."""
        logger.info("Executing configuration synchronization sequence...")
        self.merged_config = {}
        self._load_and_resolve_configs()
        self._setup_dynamic_attributes()

        if self._validate_cloud_flag:
            self.validate_configuration()

        if self._create_paths:
            self._ensure_paths_exist()

        # Resolve secrets after full initialization to avoid circular dependency
        if self._resolve_secrets_flag:
            self._resolve_secrets_in_config()

        self.feature_sets = self._generate_feature_lists()
        logger.info("Configuration successfully refreshed.")

    def _load_and_resolve_configs(self):
        """Loads and merges all YAML configuration assets found in the templates directory."""
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Scanning for configuration templates in: {self.config_dir}")
            key_sources: dict[str, str] = {}

            config_files = self.file_manager.find_files("*.yaml", search_dir=self.config_dir)
            self._process_config_files(config_files, key_sources)

            # NOTE: Skip secret resolution during initial load to avoid circular dependency
            # Secrets will be resolved after full initialization
            # self._resolve_secrets_in_config()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Configuration state synchronized. Keys: {list(self.merged_config.keys())}")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Critical synchronization failure in UnifiedConfigManager: {e}")
            raise

    def _process_config_files(self, config_files: Sequence[str | Path], key_sources: dict[str, str]):
        """Process and merge configuration files with deduplication and explicit precedence."""
        # Deduplicate config files - use set to ensure each file is processed only once
        seen_paths = set()
        processed_files = []

        for config_path in config_files:
            # Normalize the path to handle potential duplicates
            normalized_path = Path(config_path).resolve()
            if normalized_path not in seen_paths:
                seen_paths.add(normalized_path)
                processed_files.append(config_path)

        # Sort config files by explicit precedence order
        # Lower precedence files are loaded first, higher precedence files override
        processed_files = self._sort_config_by_precedence(processed_files)

        for config_path in processed_files:
            path_obj = Path(config_path)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Processing template: {path_obj}")
            config_data = self.file_manager.load_yaml(path_obj)

            if config_data:
                self._merge_config_data(config_data, path_obj, key_sources)
            else:
                logger.warning(f"Template parsing failed or file is empty: {path_obj}")

    def _sort_config_by_precedence(self, config_files: list[str | Path]) -> list[str | Path]:
        """
        Sort configuration files by explicit precedence order.
        
        Precedence (lowest to highest):
        1. Base infrastructure configs (paths, system, version) - foundational
        2. Data source configs (data_sources, collectors) - data layer
        3. Feature configs (features, transformers, enrichment) - feature layer
        4. Model configs (models, experiments, analysis) - model layer
        5. Strategy/trading configs (strategy, risk_management, simulation) - trading layer
        6. Monitoring/configs (monitoring_config, error_handling) - observability layer
        7. Override configs (unified_config) - highest precedence overrides
        
        This ensures predictable merge behavior instead of filesystem-dependent alphabetical order.
        """
        precedence_order = [
            # Layer 1: Base infrastructure (lowest precedence)
            'paths.yaml',
            'system.yaml',
            'version.yaml',
            'cloud_storage.yaml',
            # Layer 2: Data sources
            'data_sources.yaml',
            'collectors.yaml',
            # Layer 3: Features
            'features.yaml',
            'transformers.yaml',
            'enrichment.yaml',
            'noise_filter_config.yaml',
            'sentiment.yaml',
            'news_impact_classification.yaml',
            # Layer 4: Models
            'models.yaml',
            'experiments.yaml',
            'analysis.yaml',
            # Layer 5: Trading/Strategy
            'strategy.yaml',
            'risk_management.yaml',
            'simulation.yaml',
            'targets.yaml',
            # Layer 6: Monitoring/Observability
            'monitoring_config.yaml',
            'error_handling.yaml',
            'processing.yaml',
            # Layer 7: Context/Knowledge
            'context.yaml',
            'knowledge_base.yaml',
            'generated_context_rules.yaml',
            # Layer 8: Assets/Other
            'assets.yaml',
            # Layer 9: Unified override (highest precedence)
            'unified_config.yaml',
        ]

        # Create a mapping from filename to precedence index
        precedence_map = {name: idx for idx, name in enumerate(precedence_order)}

        def get_precedence(file_path: str | Path) -> int:
            """Get precedence index for a config file."""
            filename = Path(file_path).name
            # Files not in precedence map get highest precedence (loaded last)
            return precedence_map.get(filename, len(precedence_order))

        # Sort by precedence
        sorted_files = sorted(config_files, key=get_precedence)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Config files sorted by precedence: {[Path(f).name for f in sorted_files]}")

        return sorted_files

    def _merge_config_data(self, config_data: dict[str, Any], config_path: Path, key_sources: dict[str, str]):
        """Merge configuration data and track key sources."""
        for key in config_data.keys():
            self._track_key_source(key, config_path, key_sources)

        self.merged_config = _deep_merge(config_data, self.merged_config)

    def _track_key_source(self, key: str, config_path: Path, key_sources: dict[str, str]):
        """Track configuration key sources for conflict detection."""
        if key in key_sources:
            logger.warning(f"Conflicting top-level key '{key}' in {config_path.name}. "
                           f"Previous source: {key_sources[key]}. Precedence given to latest.")
        key_sources[key] = config_path.name

    def _resolve_secrets_in_config(self):
        """Resolve secrets and placeholders in configuration."""  # audit-ignore: PLACEHOLDER_SECRET_REVIEW
        secrets_manager = SecretsManager()
        all_secrets = secrets_manager.as_dict()
        self.merged_config = self._resolve_secrets_and_paths(self.merged_config, all_secrets)

    def _resolve_secrets_and_paths(self, config: Any, secrets: dict[str, str]) -> Any:
        """Recursively parses configuration for environment markers and path placeholders."""  # audit-ignore: PLACEHOLDER_SECRET_REVIEW
        if isinstance(config, dict):
            return self._resolve_dict_secrets_and_paths(config, secrets)
        elif isinstance(config, list):
            return self._resolve_list_secrets_and_paths(config, secrets)
        return config

    def _resolve_dict_secrets_and_paths(self, config: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
        """Resolve secrets and paths in dictionary configuration."""
        new_dict: dict[str, Any] = {}
        for key, value in config.items():
            new_dict[key] = self._resolve_config_value(key, value, secrets)
        return new_dict

    def _resolve_list_secrets_and_paths(self, config: list[Any], secrets: dict[str, str]) -> list[Any]:
        """Resolve secrets and paths in list configuration."""
        return [self._resolve_secrets_and_paths(item, secrets) for item in config]

    def _resolve_config_value(self, key: str, value: Any, secrets: dict[str, str]) -> Any:
        """Resolve a single configuration value."""
        if self._is_env_secret_key(key):
            return self._resolve_env_secret(key, value, secrets)
        elif self._has_placeholders(value):
            return self._resolve_placeholders(value, secrets)
        else:
            return self._resolve_secrets_and_paths(value, secrets)

    def _is_env_secret_key(self, key: str) -> bool:
        """Check if key is an environment secret key."""
        return isinstance(key, str) and key.endswith('_env')

    def _has_placeholders(self, value: Any) -> bool:
        """Check if value contains placeholders."""
        return isinstance(value, str) and '${' in value and '}' in value

    def _resolve_env_secret(self, key: str, value: Any, secrets: dict[str, str]) -> Any:
        """Resolve environment variable secret."""
        secret_key_name = value

        if secret_key_name in secrets:
            return secrets[secret_key_name]
        else:
            logger.warning(f"Credential missing: Secret key '{secret_key_name}' for '{key}' is undefined.")
            return None

    def _resolve_placeholders(self, value: str, secrets: dict[str, str]) -> Any:
        """Resolve placeholders in string value."""
        placeholders = re.findall(r'\$\{([^}]+)\}', value)
        resolved_value = value

        for placeholder in placeholders:
            resolved_placeholder = self.get(placeholder, "")
            if resolved_placeholder:
                resolved_value = resolved_value.replace(f'${{{placeholder}}}', str(resolved_placeholder))

        return self._resolve_secrets_and_paths(resolved_value, secrets)

    def _setup_dynamic_attributes(self):
        """Exposes top-level configuration keys as class attributes for cleaner access."""
        for key, value in self.merged_config.items():
            if isinstance(value, dict):
                setattr(self, key, DynamicConfig(value))
            else:
                setattr(self, key, value)

    def validate_configuration(self):
        """Enforces schema requirements for production-readiness."""
        logger.info("Executing configuration compliance audit...")
        required_sections = ['paths', 'assets', 'models', 'features', 'cloud_storage']
        for section in required_sections:
            if not self.get(section):
                raise ValueError(f"Compliance Failure: Missing mandatory configuration section '{section}'")

        # Infrastructure-critical verification
        self._validate_cloud_storage_config()

        logger.info("Configuration audit verified. System is compliant.")

    def _validate_cloud_storage_config(self):
        """Validate cloud storage configuration requirements."""
        cloud_config = self.get('cloud_storage')

        if not cloud_config:
            raise ValueError("Compliance Failure: 'cloud_storage' configuration section is missing.")

        project_id = cloud_config.get('project_id')
        bucket_name = cloud_config.get('bucket_name')

        if not project_id:
            raise ValueError("Compliance Failure: 'cloud_storage' requires 'project_id'.")

        if not bucket_name:
            raise ValueError("Compliance Failure: 'cloud_storage' requires 'bucket_name'.")

    def _ensure_paths_exist(self):
        """Synchronizes local filesystem with configured path requirements."""
        logger.info("Synchronizing filesystem structure with configuration...")
        paths_config = self.get('paths')

        if not paths_config:
            logger.info("Filesystem synchronization complete.")
            return

        paths_dict = self._get_paths_dict(paths_config)
        self._process_path_configurations(paths_dict)
        logger.info("Filesystem synchronization complete.")

    def _get_paths_dict(self, paths_config: Any) -> dict[str, Any]:
        """Get paths dictionary from configuration."""
        return paths_config.as_dict() if isinstance(paths_config, DynamicConfig) else paths_config

    def _process_path_configurations(self, paths_dict: dict[str, Any]) -> None:
        """Process all path configurations."""
        for key, path_str in paths_dict.items():
            if self._is_valid_path_string(path_str):
                self._create_path_if_needed(key, path_str)

    def _is_valid_path_string(self, path_str: Any) -> bool:
        """Check if path string is valid for processing."""
        return bool(path_str and isinstance(path_str, str))

    def _create_path_if_needed(self, key: str, path_str: str) -> None:
        """Create directory for a path if needed."""
        path_obj = self._resolve_path_object(path_str)
        dir_to_create = self._determine_directory_to_create(path_obj)

        try:
            self.file_manager.ensure_directory(dir_to_create)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"FS Integrity: Path '{key}' resolution ('{dir_to_create}') verified.")
        except OSError as e:
            logger.exception(f"FS Sync Failure for '{key}' ('{path_str}'): {e}")

    def _resolve_path_object(self, path_str: str) -> Path:
        """Resolve path string to Path object."""
        if not os.path.isabs(path_str):
            return self.project_root / path_str
        return Path(path_str)

    def _determine_directory_to_create(self, path_obj: Path) -> Path:
        """Determine which directory to create based on path object."""
        return path_obj.parent if path_obj.suffix else path_obj

    def get_runtime_params_path(self, default: str | None = None, batch_name: str | None = None) -> Path:
        """
        Retrieves the authoritative path for runtime parameter artifacts.

        Priority:
        1. Batch-specific path: outputs/{batch_name}/runtime_params.json
        2. Default path from config
        3. Legacy fallback
        """
        # Priority 1: Batch-specific path
        if batch_name:
            batch_path = self.project_root / 'outputs' / batch_name / 'runtime_params.json'
            if batch_path.exists():
                return batch_path

        # Priority 2: Default path from config
        runtime_val = default or self.get('paths.runtime_params', None)
        if not runtime_val:
            runtime_val = self.get('system.runtime_params_path', 'data/runtime/runtime_params.json')

        runtime_path: Path
        if isinstance(runtime_val, str) and not os.path.isabs(runtime_val):
            runtime_path = self.project_root / runtime_val
        else:
            runtime_path = Path(runtime_val)

        if runtime_path.exists():
            return runtime_path

        # Priority 3: Legacy compatibility fallback
        fallback = self.project_root / 'src' / 'config' / 'runtime_params.json'
        if fallback.exists():
            return fallback

        return runtime_path

    def get_models_path(self) -> Path:
        """Resolves the standard storage location for serialized models."""
        models_path = self.get('paths.models', None)
        if not models_path:
            models_path = self.get('system.models_path', 'data/trained_models')

        return self._resolve_output_path(models_path)

    def get_cache_path(self) -> Path:
        """Resolves the standardized directory for feature and inference caching."""
        cache_path = self.get('paths.cache', 'data/cache')
        return self._resolve_output_path(cache_path)

    def get_selected_features_cache_path(self) -> Path:
        """Resolves the specific artifact path for optimized feature selections."""
        cache_dir = self.get_cache_path()
        return cache_dir / 'selected_features.json'

    def get_accumulation_output_dir(self) -> Path:
        """Resolves the destination directory for batch data accumulation tasks."""
        output_dir = self.get('system.accumulation.output_dir', 'data/colab/accumulated')
        return self._resolve_output_path(output_dir)

    def _resolve_output_path(self, output_dir: Any) -> Path:
        """Resolve output path to absolute Path object."""
        res_path: Path
        if isinstance(output_dir, str) and not os.path.isabs(output_dir):
            res_path = self.project_root / output_dir
        else:
            res_path = Path(output_dir)
        return res_path

    def _generate_feature_lists(self) -> dict[str, list[str]]:
        """Internal generator for logical feature groupings."""
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Hierarchical accessor for configuration keys.
        Supports dotted notation (e.g., 'paths.models').
        """
        value = self._traverse_nested_keys(key, default)
        if value is not default:
            value = self._resolve_placeholders_in_value(value)
        return value

    def get_specific_config(self, section: str, subsection: str | None = None, default: Any = None) -> Any:
        """
        Two-level nested accessor (e.g. get_specific_config('strategy', 'backtesting')).

        Backward-compatible shim kept for call sites that predate dotted
        notation. Equivalent to get(f'{section}.{subsection}') when a
        subsection is given, otherwise get(section). Returns ``default``
        (None by default) when the path is absent.
        """
        key = f"{section}.{subsection}" if subsection else section
        value = self.get(key, default)
        if value is None:
            return default
        return value

    def _traverse_nested_keys(self, key: str, default: Any) -> Any:
        """Traverse nested keys using dotted notation."""
        keys = key.split('.')
        value = self.merged_config

        for k in keys:
            value = self._get_nested_value(value, k, default)
            if value is default:
                return default

        return value

    def _get_nested_value(self, value: Any, key: str, default: Any) -> Any:
        """Get value from nested structure."""
        if isinstance(value, dict) and key in value:
            return value[key]
        elif isinstance(value, DynamicConfig) and key in value._data:
            return value._data[key]
        else:
            return default

    def _resolve_placeholders_in_value(self, value: Any) -> Any:
        """Resolve placeholders in string value."""
        if not isinstance(value, str) or '${' not in value:
            return value

        placeholders = re.findall(r'\$\{([^}]+)\}', value)
        for placeholder in placeholders:
            resolved_placeholder = self.get(placeholder, "")
            if not isinstance(resolved_placeholder, (str, int, float, bool)):
                resolved_placeholder = str(resolved_placeholder)
            value = value.replace(f'${{{placeholder}}}', str(resolved_placeholder))

        return value

    #: Sections asked for but absent, reported once each (see get_config).
    _absent_sections_reported: ClassVar[set[str]] = set()

    def get_config(self, name: str, default: Any = None) -> Any:
        """Access a top-level configuration section.

        `name` is a TOP-LEVEL YAML KEY across the merged config, not a file
        name: `get_config('processing')` is None even though processing.yaml
        exists, because its top-level keys are `safe_fill` and
        `data_preparation`.

        A missing section is reported ONCE at warning level. It used to return
        None in silence, and callers almost always write `or {}` immediately
        after, so an absent section degrades into code defaults with nothing
        said. An audit of every `get_config(...)` call in the codebase found
        10 of 24 requested keys did not exist -- among them `processing`,
        which left IntelligentDataFilter running unconfigured on every run,
        and `modeling`, which made four training settings unreachable.
        """
        found = name in self.merged_config
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Direct template access attempt for: '{name}'. Found: {found}")
        if not found and name not in UnifiedConfigManager._absent_sections_reported:
            UnifiedConfigManager._absent_sections_reported.add(name)
            logger.warning(
                f"Config section '{name}' does not exist in any YAML; the "
                f"caller will fall back to its built-in defaults. Add the "
                f"section, or drop the lookup if it is obsolete."
            )
        return self.merged_config.get(name, default)


_config_instance: UnifiedConfigManager | None = None
_config_lock = threading.Lock()


def get_current_config(config_dir: str | None = None, force_reload: bool = False) -> UnifiedConfigManager:
    """
    Standard thread-safe singleton factory for the UnifiedConfigManager interface.
    Utilizes double-checked locking for optimized initial concurrency.

    Args:
        config_dir: Directory containing YAML configuration templates.
                    If provided and differs from existing instance, a warning is logged.
                    Use force_reload=True to reinitialize with a different config_dir.
        force_reload: If True, forces reinitialization even if instance exists.
                     Useful for loading different configurations in tests.

    Returns:
        The singleton UnifiedConfigManager instance.
    """
    global _config_instance

    # Fast path: instance already initialized
    if _config_instance is not None and not force_reload:
        # Check if config_dir differs from existing instance
        if config_dir is not None:
            existing_config_dir = str(_config_instance.config_dir)
            requested_config_dir = str(Path(config_dir).resolve())
            if existing_config_dir != requested_config_dir:
                logger.warning(
                    f"Config directory mismatch: existing='{existing_config_dir}', "
                    f"requested='{requested_config_dir}'. Using existing instance. "
                    f"Set force_reload=True to reinitialize with new config_dir."
                )
        return _config_instance

    # Protected path: locked initialization
    with _config_lock:
        # Re-verify instance in case of race condition during wait
        if _config_instance is not None and not force_reload:
            if config_dir is not None:
                existing_config_dir = str(_config_instance.config_dir)
                requested_config_dir = str(Path(config_dir).resolve())
                if existing_config_dir != requested_config_dir:
                    logger.warning(
                        f"Config directory mismatch: existing='{existing_config_dir}', "
                        f"requested='{requested_config_dir}'. Using existing instance. "
                        f"Set force_reload=True to reinitialize with new config_dir."
                    )
            return _config_instance

        # Establish operational context
        effective_config_dir = config_dir or str(Path(__file__).parent)
        env_str = os.getenv('TRADING_ENV', Environment.DEVELOPMENT.value).lower()
        try:
            env = Environment(env_str)
        except ValueError:
            logger.warning(f"Unrecognized TRADING_ENV state '{env_str}'. Defaulting to development protocol.")
            env = Environment.DEVELOPMENT

        if force_reload and _config_instance is not None:
            logger.info(f"Force reloading configuration from: {effective_config_dir}")

        _config_instance = UnifiedConfigManager(env, effective_config_dir)
        return _config_instance
