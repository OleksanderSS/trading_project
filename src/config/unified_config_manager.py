import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

# Оновлені імпорти
from src.core.logging.logger import ProjectLogger
from src.core.security.secure_secrets_manager import SecretsManager
from src.core.file_management.file_manager import FileManager

logger = ProjectLogger.get_logger("UnifiedConfigManager")

class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class DynamicConfig:
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DynamicConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def as_dict(self) -> Dict[str, Any]:
        return self._data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return DynamicConfig(value)
            return value
        raise AttributeError(f"\'{type(self).__name__}\' object has no attribute \'{name}\'s")

    def __repr__(self) -> str:
        return str(self._data)

def _deep_merge(source: Dict, destination: Dict) -> Dict:
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

class UnifiedConfigManager:
    def __init__(self, env: Environment = Environment.DEVELOPMENT, config_dir: Optional[str] = None):
        self.env = env
        if config_dir:
            self.config_dir = Path(config_dir).resolve()
        else:
            # Правильний шлях до папки конфігурації
            self.config_dir = Path(__file__).parent 
        
        self.project_root = self.config_dir.parent.parent
        self.file_manager = FileManager(base_dir=self.project_root)
        self.merged_config: Dict[str, Any] = {}
        self.feature_sets: Dict[str, List[str]] = {}

        self._load_and_resolve_configs()
        self._setup_dynamic_attributes()
        
        self.validate_configuration()
        self._ensure_paths_exist()
        
        self.feature_sets = self._generate_feature_lists()

        logger.info(f"[UnifiedConfigManager] Initialized for \'{self.env.value}\' environment. Configuration loaded.")

    def reload(self):
        """Refreshes the configuration without restarting the application."""
        logger.info("Reloading configuration...")
        self.merged_config = {}
        self._load_and_resolve_configs()
        self._setup_dynamic_attributes()
        self.validate_configuration()
        self._ensure_paths_exist()
        self.feature_sets = self._generate_feature_lists()
        logger.info("Configuration reloaded successfully.")

    def _load_and_resolve_configs(self):
        try:
            logger.info(f"Loading configurations from: {self.config_dir}")
            key_sources = {}
            
            # Використовуємо FileManager для пошуку та завантаження конфігів
            config_files = self.file_manager.find_files("*.yaml", search_dir=self.config_dir)
            for config_path in config_files:
                logger.debug(f"Loading config file: {config_path}")
                config_data = self.file_manager.load_yaml(config_path)
                
                if config_data:
                    for key in config_data.keys():
                        if key in key_sources:
                            logger.warning(f"Duplicate top-level key \'{key}\' found in {config_path.name}. "
                                           f"Previously defined in {key_sources[key]}. Overwriting.")
                        key_sources[key] = config_path.name
                    
                    logger.debug(f"Parsed data from {config_path}: {list(config_data.keys())}")
                    self.merged_config = _deep_merge(config_data, self.merged_config)
                else:
                    logger.warning(f"File {config_path} is empty, invalid YAML, or failed to load. Skipping.")

            logger.debug(f"Initial merged configuration keys: {list(self.merged_config.keys())}")

            # Вирішуємо секрети та шляхи
            secrets_manager = SecretsManager()
            all_secrets = secrets_manager.as_dict()
            self.merged_config = self._resolve_secrets_and_paths(self.merged_config, all_secrets)
            logger.debug(f"Final merged configuration keys: {list(self.merged_config.keys())}")

        except Exception as e:
            logger.error(f"[UnifiedConfigManager] Critical error loading configurations: {e}", exc_info=True)
            raise

    def _resolve_secrets_and_paths(self, config: Any, secrets: Dict[str, str]) -> Any:
        if isinstance(config, dict):
            new_dict = {}
            for key, value in config.items():
                if isinstance(key, str) and key.endswith('_env'):
                    secret_key_name = value
                    new_key_name = key[:-4]
                    
                    if secret_key_name in secrets:
                        new_dict[new_key_name] = secrets[secret_key_name]
                    else:
                        new_dict[new_key_name] = None
                        logger.warning(f"Secret \'{secret_key_name}\' for key \'{key}\' not found in SecretsManager.")
                elif isinstance(value, str) and '${' in value and '}' in value:
                    placeholders = re.findall(r'\$\{([^}]+)\}', value)
                    resolved_value = value
                    for placeholder in placeholders:
                        resolved_placeholder = self.get(placeholder, "")
                        if resolved_placeholder:
                             resolved_value = resolved_value.replace(f'${{{placeholder}}}', str(resolved_placeholder))
                    new_dict[key] = self._resolve_secrets_and_paths(resolved_value, secrets)
                else:
                    new_dict[key] = self._resolve_secrets_and_paths(value, secrets)
            return new_dict
        elif isinstance(config, list):
            return [self._resolve_secrets_and_paths(item, secrets) for item in config]
        return config

    def _setup_dynamic_attributes(self):
        for key, value in self.merged_config.items():
            if isinstance(value, dict):
                setattr(self, key, DynamicConfig(value))
            else:
                setattr(self, key, value)

    def validate_configuration(self):
        logger.info("Validating configuration...")
        # Оновлений список для перевірки
        required_sections = ['paths', 'assets', 'models', 'features', 'cloud_storage']
        for section in required_sections:
            if not self.get(section):
                raise ValueError(f"Configuration validation failed: Missing required section \'{section}\'")
        
        # Перевірка наявності ключових полів у секції cloud_storage
        cloud_config = self.get('cloud_storage')
        if not cloud_config or not cloud_config.get('project_id') or not cloud_config.get('bucket_name'):
            raise ValueError("Configuration validation failed: 'cloud_storage' section must contain 'project_id' and 'bucket_name'.")

        logger.info("Configuration validation successful.")

    def _ensure_paths_exist(self):
        logger.info("Ensuring all configured paths exist...")
        paths_config = self.get('paths')
        if paths_config:
            paths_dict = paths_config.as_dict() if isinstance(paths_config, DynamicConfig) else paths_config
            for key, path_str in paths_dict.items():
                if path_str and isinstance(path_str, str):
                    if not os.path.isabs(path_str):
                        path_obj = self.project_root / path_str
                    else:
                        path_obj = Path(path_str)
                    
                    try:
                        # If path has a suffix, it's a file, so ensure parent directory exists
                        dir_to_create = path_obj.parent if path_obj.suffix else path_obj
                        self.file_manager.ensure_directory(dir_to_create)
                        logger.debug(f"Path for '{key}' ('{dir_to_create}') exists or was created.")
                    except OSError as e:
                        logger.error(f"Failed to process path for '{key}' ('{path_str}'): {e}", exc_info=True)
        logger.info("Path check complete.")

    def _generate_feature_lists(self) -> Dict[str, List[str]]:
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.merged_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            elif isinstance(value, DynamicConfig) and k in value._data:
                value = value._data[k]
            else:
                return default

        if isinstance(value, str) and '${' in value:
            placeholders = re.findall(r'\$\{([^}]+)\}', value)
            for placeholder in placeholders:
                resolved_placeholder = self.get(placeholder, "")
                if not isinstance(resolved_placeholder, (str, int, float, bool)):
                    resolved_placeholder = str(resolved_placeholder)
                value = value.replace(f'${{{placeholder}}}', str(resolved_placeholder))

        if isinstance(value, dict):
            return DynamicConfig(value)
        return value
    
    def get_config(self, name: str, default: Any = None) -> Any:
        logger.debug(f"Getting config for key: \'{name}\'. Found: {name in self.merged_config}")
        return self.merged_config.get(name, default)

_config_instance: Optional["UnifiedConfigManager"] = None

def get_current_config(config_dir: Optional[str] = None) -> "UnifiedConfigManager":
    global _config_instance
    if _config_instance is None:
        # Передаємо правильний шлях до папки з конфігураціями
        effective_config_dir = config_dir or str(Path(__file__).parent)
        env_str = os.getenv('TRADING_ENV', Environment.DEVELOPMENT.value).lower()
        try:
            env = Environment(env_str)
        except ValueError:
            logger.warning(f"Invalid TRADING_ENV value \'{env_str}\'. Falling back to development.")
            env = Environment.DEVELOPMENT
        _config_instance = UnifiedConfigManager(env, effective_config_dir)
    return _config_instance
