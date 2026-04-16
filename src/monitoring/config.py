"""
Monitoring Configuration - Конфігурація системи моніторингу.

Містить:
- Конфігурації для всіх моніторів
- Налаштування сповіщень
- Налаштування дашборду
- Приклади конфігурацій для різних середовищ

Використовує:
- YAML-подібний формат
- Середовищні змінні
- Валідація конфігурації
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

class MonitoringConfig:
    """Клас для управління конфігурацією моніторингу"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.config = {}
        self.load_config()

    def _get_default_config_path(self) -> str:
        """Отримання шляху до конфігураційного файлу за замовчуванням"""
        return os.path.join(os.path.dirname(__file__), 'monitoring_config.yaml')

    def load_config(self):
        """Завантаження конфігурації"""
        try:
            if os.path.exists(self.config_file):
                import yaml
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
            self.config = self._get_default_config()

        # Застосування змінних середовища
        self._apply_environment_variables()

        # Валідація конфігурації
        self._validate_config()

    def save_config(self):
        """Збереження конфігурації"""
        try:
            import yaml
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving config: {e}")

    def _apply_environment_variables(self):
        """Застосування змінних середовища до конфігурації"""
        # Системні пороги
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
        """Парсинг значення змінної середовища"""
        # Спроба конвертації в число
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Булеві значення
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        return value

    def _set_nested_config(self, config: Dict[str, Any], path: tuple, value: Any):
        """Встановлення значення в вкладеній конфігурації"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _validate_config(self):
        """Валідація конфігурації"""
        # Валідація collection_interval
        interval = self.config.get('collection_interval', 30)
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError("collection_interval must be a positive number")

        # Валідація порогових значень
        system_health = self.config.get('system_health', {})
        for threshold in ['cpu_threshold', 'memory_threshold', 'disk_threshold']:
            value = system_health.get(threshold, 80)
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"{threshold} must be between 0 and 100")

        # Валідація каналів сповіщень
        alerts = self.config.get('alerts', {})
        channels = alerts.get('channels', ['log'])
        valid_channels = ['log', 'email', 'slack']
        for channel in channels:
            if channel not in valid_channels:
                raise ValueError(f"Invalid alert channel: {channel}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Отримання конфігурації за замовчуванням"""
        return {
            # Основні налаштування
            'collection_interval': 30,  # секунди
            'enabled_monitors': ['system_health', 'model_performance', 'data_quality'],

            # Моніторинг системного здоров'я
            'system_health': {
                'cpu_threshold': 80.0,      # %
                'memory_threshold': 85.0,   # %
                'disk_threshold': 90.0,     # %
                'network_timeout': 30,      # секунди
                'history_size': 100,        # кількість записів
                'enabled_metrics': ['cpu', 'memory', 'disk', 'network', 'processes']
            },

            # Моніторинг продуктивності моделей
            'model_performance': {
                'accuracy_threshold': 0.7,    # мінімальна точність
                'mae_threshold': 0.1,         # максимальна MAE
                'drift_threshold': 0.05,      # поріг дрейфу
                'performance_window': 100,    # вікно для розрахунку продуктивності
                'baseline_period_days': 30,   # період для baseline
                'enabled_metrics': ['accuracy', 'mae', 'drift', 'latency']
            },

            # Моніторинг якості даних
            'data_quality': {
                'missing_threshold': 0.05,      # 5% пропущених значень
                'outlier_threshold': 0.1,       # 10% викидів
                'consistency_threshold': 0.95,  # 95% консистентності
                'duplicate_threshold': 0.02,    # 2% дублікатів
                'enabled_checks': ['completeness', 'consistency', 'outliers', 'duplicates']
            },

            # Система сповіщень
            'alerts': {
                'channels': ['log'],                    # доступні канали
                'auto_resolve_hours': 24,              # авто-вирішення через години
                'max_alerts_per_hour': 10,             # максимум сповіщень за годину
                'severity_levels': ['info', 'warning', 'error', 'critical'],
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': os.getenv('MONITORING_EMAIL_USER'),
                    'password': os.getenv('MONITORING_EMAIL_PASS'),
                    'recipients': ['admin@example.com']
                },
                'slack': {
                    'webhook_url': os.getenv('MONITORING_SLACK_WEBHOOK'),
                    'channel': '#monitoring',
                    'username': 'Trading Monitor'
                }
            },

            # Дашборд
            'dashboard': {
                'refresh_interval': 5000,      # ms
                'history_days': 7,            # дні історії
                'auto_save': True,            # автоматичне збереження звітів
                'save_interval': 3600,        # секунди
                'save_path': 'monitoring_reports',
                'web': {
                    'port': 8050,
                    'host': 'localhost',
                    'debug': False,
                    'update_interval': 5000
                },
                'text': {
                    'max_lines': 1000,
                    'include_timestamps': True
                }
            },

            # Логування
            'logging': {
                'level': 'INFO',
                'file': 'monitoring.log',
                'max_file_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            },

            # Продуктивність
            'performance': {
                'max_workers': 4,
                'queue_size': 1000,
                'batch_size': 100,
                'cache_size': 1000
            }
        }

    def get_config_for_environment(self, environment: str) -> Dict[str, Any]:
        """Отримання конфігурації для конкретного середовища"""
        base_config = self.config.copy()

        if environment == 'development':
            base_config.update({
                'collection_interval': 10,
                'system_health': {
                    'cpu_threshold': 90.0,
                    'memory_threshold': 90.0,
                    'disk_threshold': 95.0
                },
                'dashboard': {
                    'web': {'debug': True}
                }
            })
        elif environment == 'staging':
            base_config.update({
                'collection_interval': 20,
                'alerts': {
                    'channels': ['log', 'email']
                }
            })
        elif environment == 'production':
            base_config.update({
                'collection_interval': 60,
                'alerts': {
                    'channels': ['log', 'email', 'slack']
                },
                'dashboard': {
                    'auto_save': True,
                    'save_interval': 1800  # 30 хвилин
                }
            })

        return base_config

# Глобальна конфігурація
_default_config = None

def get_monitoring_config(config_file: Optional[str] = None) -> MonitoringConfig:
    """Отримання екземпляру конфігурації моніторингу"""
    global _default_config
    if _default_config is None:
        _default_config = MonitoringConfig(config_file)
    return _default_config

def create_config_file(filepath: str, environment: str = 'development'):
    """Створення конфігураційного файлу"""
    config = MonitoringConfig()
    env_config = config.get_config_for_environment(environment)

    # Тимчасово змінюємо конфігурацію
    original_config = config.config
    config.config = env_config

    try:
        config.config_file = filepath
        config.save_config()
        print(f"Configuration file created: {filepath}")
    finally:
        config.config = original_config

# Приклад використання
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitoring Configuration')
    parser.add_argument('--create-config', help='Create configuration file')
    parser.add_argument('--environment', default='development',
                       choices=['development', 'staging', 'production'],
                       help='Environment for configuration')

    args = parser.parse_args()

    if args.create_config:
        create_config_file(args.create_config, args.environment)
    else:
        # Виведення поточної конфігурації
        config = get_monitoring_config()
        import json
        print(json.dumps(config.config, indent=2, ensure_ascii=False))