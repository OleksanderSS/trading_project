"""
Monitoring Configuration - Monitoring system configuration.

Contains configurations for all monitors, alert settings, and dashboard.
"""

import os
from typing import Any

from src.core.logging.logger import ProjectLogger  # type: ignore


class MonitoringConfig:
    """Class for managing monitoring configuration"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or self._get_default_config_path()
        self.config = {}
        self.logger = ProjectLogger.get_logger("MonitoringConfig")
        self.load_config()

    def _get_default_config_path(self) -> str:
        """Getting default config file path"""
        return os.path.join(os.path.dirname(__file__), 'monitoring_config.yaml')

    def load_config(self):
        """Loading configuration"""
        try:
            if os.path.exists(self.config_file):
                import yaml
                with open(self.config_file, encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            self.logger.warning(f"Could not load config file {self.config_file}: {e}")
            self.config = self._get_default_config()

        # Applying environment variables
        self._apply_environment_variables()

        # Configuration validation
        self._validate_config()

    def save_config(self):
        """Saving configuration"""
        try:
            import yaml
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def _apply_environment_variables(self):
        """Apply environment variables to config"""
        env_mappings = {
            'MONITORING_CPU_THRESHOLD': ('system_health', 'cpu_threshold'),
            'MONITORING_MEMORY_THRESHOLD': ('system_health', 'memory_threshold'),
            'MONITORING_DISK_THRESHOLD': ('system_health', 'disk_threshold'),
            'MONITORING_COLLECTION_INTERVAL': ('collection_interval',),
            'MONITORING_AUTO_RESOLVE_HOURS': ('alerts', 'auto_resolve_hours'),
            'MONITORING_DASHBOARD_PORT': ('dashboard', 'web', 'port'),
            'MONITORING_DASHBOARD_HOST': ('dashboard', 'web', 'host'),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_config(self.config, config_path, self._parse_env_value(value))

    def _parse_env_value(self, value: str):
        try:
            if '.' in value: return float(value)
            return int(value)
        except ValueError: pass
        if value.lower() in ('true', 'yes', '1'): return True
        if value.lower() in ('false', 'no', '0'): return False
        return value

    def _set_nested_config(self, config: dict[str, Any], path: tuple, value: Any):
        current = config
        for key in path[:-1]:
            if key not in current: current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _validate_config(self):
        interval = self.config.get('collection_interval', 30)
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError("collection_interval must be a positive number")
        system_health = self.config.get('system_health', {})
        for threshold in ['cpu_threshold', 'memory_threshold', 'disk_threshold']:
            value = system_health.get(threshold, 80)
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"{threshold} must be between 0 and 100")

    def _get_default_config(self) -> dict[str, Any]:
        return {
            'collection_interval': 30,
            'enabled_monitors': ['system_health', 'model_performance', 'data_quality'],
            'system_health': {
                'cpu_threshold': 80.0, 'memory_threshold': 85.0, 'disk_threshold': 90.0,
                'network_timeout': 30, 'history_size': 100,
                'enabled_metrics': ['cpu', 'memory', 'disk', 'network', 'processes']
            },
            'model_performance': {
                'accuracy_threshold': 0.7, 'mae_threshold': 0.1, 'drift_threshold': 0.05,
                'performance_window': 100, 'baseline_period_days': 30,
                'enabled_metrics': ['accuracy', 'mae', 'drift', 'latency']
            },
            'data_quality': {
                'missing_threshold': 0.05, 'outlier_threshold': 0.1, 'consistency_threshold': 0.95,
                'duplicate_threshold': 0.02,
                'enabled_checks': ['completeness', 'consistency', 'outliers', 'duplicates']
            },
            'alerts': {
                'channels': ['log'], 'auto_resolve_hours': 24, 'max_alerts_per_hour': 10,
                'severity_levels': ['info', 'warning', 'error', 'critical']
            },
            'dashboard': {
                'refresh_interval': 5000, 'history_days': 7, 'auto_save': True,
                'save_interval': 3600, 'save_path': 'monitoring_reports',
                'web': {'port': 8050, 'host': 'localhost', 'debug': False, 'update_interval': 5000}
            }
        }

    def get_config_for_environment(self, environment: str) -> dict[str, Any]:
        base_config = self.config.copy()
        if environment == 'development':
            base_config.update({'collection_interval': 10})
        elif environment == 'staging':
            base_config.update({'collection_interval': 20, 'alerts': {'channels': ['log', 'email']}})
        elif environment == 'production':
            base_config.update({'collection_interval': 60, 'alerts': {'channels': ['log', 'email', 'slack']}})
        return base_config

# Global configuration
_default_config = None

def get_monitoring_config(config_file: str | None = None) -> MonitoringConfig:
    global _default_config
    if _default_config is None:
        _default_config = MonitoringConfig(config_file)
    return _default_config

def create_config_file(filepath: str, environment: str = 'development'):
    config = MonitoringConfig()
    env_config = config.get_config_for_environment(environment)
    original_config = config.config
    config.config = env_config
    try:
        config.config_file = filepath
        config.save_config()
        config.logger.info(f"Configuration file created: {filepath}")
    finally:
        config.config = original_config

if __name__ == '__main__':
    ProjectLogger.setup_logging()
    logger = ProjectLogger.get_logger("ConfigManagerRunner")
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Monitoring Configuration')
    parser.add_argument('--create-config', help='Create configuration file')
    parser.add_argument('--environment', default='development', choices=['development', 'staging', 'production'])

    args = parser.parse_args()
    if args.create_config:
        create_config_file(args.create_config, args.environment)
    else:
        cfg = get_monitoring_config()
        logger.info(f"Current Monitoring Config:\n{json.dumps(cfg.config, indent=2, ensure_ascii=False)}")
