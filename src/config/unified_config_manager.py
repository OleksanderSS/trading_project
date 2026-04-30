# src/config/unified_config_manager.py
"""
Unified Configuration Manager - Centralized hierarchical configuration system.
Supports environment-based overrides, secret resolution, and dynamic attribute access.
"""

import os
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

from src.core.logging.logger import ProjectLogger
from src.core.security.secure_secrets_manager import SecretsManager
from src.core.file_management.file_manager import FileManager

logger = ProjectLogger.get_logger("UnifiedConfigManager")

class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class DynamicConfig:
    """Wrapper for recursive attribute-selection from dictionary-based configuration."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DynamicConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the internal dictionary."""
        return self._data.get(key, default)
    
    def as_dict(self) -> Dict[str, Any]:
        """Exports content back to a standard dictionary."""
        return self._data

    def __getattr__(self, name: str) -> Any:
        value = self._get_attribute_value(name)
        if value is not None:
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _get_attribute_value(self, name: str) -> Any:
        """Get attribute value from internal data."""
        if name not in self._data:
            return None
        
        value = self._data[name]
        return DynamicConfig(value) if isinstance(value, dict) else value

    def __repr__(self) -> str:
        return str(self._data)

def _deep_merge(source: Dict, destination: Dict) -> Dict:
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
    
    def __init__(self, env: Environment = Environment.DEVELOPMENT, config_dir: Optional[str] = None):
        """
        Initializes the manager.
        
        Args:
            env: Active deployment environment.
            config_dir: Directory containing YAML configuration templates.
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
        self.merged_config: Dict[str, Any] = {}
        self.feature_sets: Dict[str, List[str]] = {}

        # Initialization sequence
        self._load_and_resolve_configs()
        self._setup_dynamic_attributes()
        
        self.validate_configuration()
        self._ensure_paths_exist()
        
        self.feature_sets = self._generate_feature_lists()

        logger.info(f"UnifiedConfigManager initialized for '{self.env.value}' environment.")

    def reload(self):
        """Refreshes the configuration from disk without process restart."""
        logger.info("Executing configuration synchronization sequence...")
        self.merged_config = {}
        self._load_and_resolve_configs()
        self._setup_dynamic_attributes()
        self.validate_configuration()
        self._ensure_paths_exist()
        self.feature_sets = self._generate_feature_lists()
        logger.info("Configuration successfully refreshed.")

    def _load_and_resolve_configs(self):
        """Loads and merges all YAML configuration assets found in the templates directory."""
        try:
            logger.debug(f"Scanning for configuration templates in: {self.config_dir}")
            key_sources = {}
            
            config_files = self.file_manager.find_files("*.yaml", search_dir=self.config_dir)
            self._process_config_files(config_files, key_sources)
            self._resolve_secrets_in_config()
            
            logger.debug(f"Configuration state synchronized. Keys: {list(self.merged_config.keys())}")

        except Exception as e:
            logger.error(f"Critical synchronization failure in UnifiedConfigManager: {e}")
            raise
    
    def _process_config_files(self, config_files: List, key_sources: Dict):
        """Process and merge configuration files with deduplication."""
        # Deduplicate config files - use set to ensure each file is processed only once
        seen_paths = set()
        processed_files = []
        
        for config_path in config_files:
            # Normalize the path to handle potential duplicates
            normalized_path = Path(config_path).resolve()
            if normalized_path not in seen_paths:
                seen_paths.add(normalized_path)
                processed_files.append(config_path)
        
        for config_path in processed_files:
            logger.debug(f"Processing template: {config_path}")
            config_data = self.file_manager.load_yaml(config_path)
            
            if config_data:
                self._merge_config_data(config_data, config_path, key_sources)
            else:
                logger.warning(f"Template parsing failed or file is empty: {config_path}")
    
    def _merge_config_data(self, config_data: Dict, config_path: Path, key_sources: Dict):
        """Merge configuration data and track key sources."""
        for key in config_data.keys():
            self._track_key_source(key, config_path, key_sources)
        
        self.merged_config = _deep_merge(config_data, self.merged_config)
    
    def _track_key_source(self, key: str, config_path: Path, key_sources: Dict):
        """Track configuration key sources for conflict detection."""
        if key in key_sources:
            logger.warning(f"Conflicting top-level key '{key}' in {config_path.name}. "
                           f"Previous source: {key_sources[key]}. Precedence given to latest.")
        key_sources[key] = config_path.name
    
    def _resolve_secrets_in_config(self):
        """Resolve secrets and placeholders in configuration."""
        secrets_manager = SecretsManager()
        all_secrets = secrets_manager.as_dict()
        self.merged_config = self._resolve_secrets_and_paths(self.merged_config, all_secrets)

    def _resolve_secrets_and_paths(self, config: Any, secrets: Dict[str, str]) -> Any:
        """Recursively parses configuration for environment markers and path placeholders."""
        if isinstance(config, dict):
            return self._resolve_dict_secrets_and_paths(config, secrets)
        elif isinstance(config, list):
            return self._resolve_list_secrets_and_paths(config, secrets)
        return config
    
    def _resolve_dict_secrets_and_paths(self, config: Dict, secrets: Dict[str, str]) -> Dict:
        """Resolve secrets and paths in dictionary configuration."""
        new_dict = {}
        for key, value in config.items():
            new_dict[key] = self._resolve_config_value(key, value, secrets)
        return new_dict
    
    def _resolve_list_secrets_and_paths(self, config: List, secrets: Dict[str, str]) -> List:
        """Resolve secrets and paths in list configuration."""
        return [self._resolve_secrets_and_paths(item, secrets) for item in config]
    
    def _resolve_config_value(self, key: str, value: Any, secrets: Dict[str, str]) -> Any:
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
    
    def _resolve_env_secret(self, key: str, value: Any, secrets: Dict[str, str]) -> Any:
        """Resolve environment variable secret."""
        secret_key_name = value
        
        if secret_key_name in secrets:
            return secrets[secret_key_name]
        else:
            logger.warning(f"Credential missing: Secret key '{secret_key_name}' for '{key}' is undefined.")
            return None
    
    def _resolve_placeholders(self, value: str, secrets: Dict[str, str]) -> Any:
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
    
    def _get_paths_dict(self, paths_config) -> Dict:
        """Get paths dictionary from configuration."""
        return paths_config.as_dict() if isinstance(paths_config, DynamicConfig) else paths_config
    
    def _process_path_configurations(self, paths_dict: Dict) -> None:
        """Process all path configurations."""
        for key, path_str in paths_dict.items():
            if self._is_valid_path_string(path_str):
                self._create_path_if_needed(key, path_str)
    
    def _is_valid_path_string(self, path_str: Any) -> bool:
        """Check if path string is valid for processing."""
        return path_str and isinstance(path_str, str)
    
    def _create_path_if_needed(self, key: str, path_str: str) -> None:
        """Create directory for a path if needed."""
        path_obj = self._resolve_path_object(path_str)
        dir_to_create = self._determine_directory_to_create(path_obj)
        
        try:
            self.file_manager.ensure_directory(dir_to_create)
            logger.debug(f"FS Integrity: Path '{key}' resolution ('{dir_to_create}') verified.")
        except OSError as e:
            logger.error(f"FS Sync Failure for '{key}' ('{path_str}'): {e}")
    
    def _resolve_path_object(self, path_str: str) -> Path:
        """Resolve path string to Path object."""
        if not os.path.isabs(path_str):
            return self.project_root / path_str
        return Path(path_str)
    
    def _determine_directory_to_create(self, path_obj: Path) -> Path:
        """Determine which directory to create based on path object."""
        return path_obj.parent if path_obj.suffix else path_obj

    def get_runtime_params_path(self, default: Optional[str] = None, batch_name: Optional[str] = None) -> Path:
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
        runtime_path = default or self.get('paths.runtime_params', None)
        if not runtime_path:
            runtime_path = self.get('system.runtime_params_path', 'data/runtime/runtime_params.json')

        if isinstance(runtime_path, str) and not os.path.isabs(runtime_path):
            runtime_path = self.project_root / runtime_path
        else:
            runtime_path = Path(runtime_path)

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
        if isinstance(output_dir, str) and not os.path.isabs(output_dir):
            return self.project_root / output_dir
        return Path(output_dir)

    def _generate_feature_lists(self) -> Dict[str, List[str]]:
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
    
    def get_config(self, name: str, default: Any = None) -> Any:
        """Legacy access interface for direct configuration segments."""
        logger.debug(f"Direct template access attempt for: '{name}'. Found: {name in self.merged_config}")
        return self.merged_config.get(name, default)

_config_instance: Optional["UnifiedConfigManager"] = None
_config_lock = threading.Lock()

def get_current_config(config_dir: Optional[str] = None) -> "UnifiedConfigManager":
    """
    Standard thread-safe singleton factory for the UnifiedConfigManager interface.
    Utilizes double-checked locking for optimized initial concurrency.
    """
    global _config_instance
    
    # Fast path: instance already initialized
    if _config_instance is not None:
        return _config_instance
    
    # Protected path: locked initialization
    with _config_lock:
        # Re-verify instance in case of race condition during wait
        if _config_instance is not None:
            return _config_instance
        
        # Establish operational context
        effective_config_dir = config_dir or str(Path(__file__).parent)
        env_str = os.getenv('TRADING_ENV', Environment.DEVELOPMENT.value).lower()
        try:
            env = Environment(env_str)
        except ValueError:
            logger.warning(f"Unrecognized TRADING_ENV state '{env_str}'. Defaulting to development protocol.")
            env = Environment.DEVELOPMENT
        
        _config_instance = UnifiedConfigManager(env, effective_config_dir)
        return _config_instance
