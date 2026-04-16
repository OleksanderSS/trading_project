"""
Monitoring Package - Пакет системи моніторингу.

Містить:
- MonitoringSystem: Головна система моніторингу
- SystemHealthMonitor: Моніторинг системного здоров'я
- ModelPerformanceMonitor: Моніторинг продуктивності моделей
- DataQualityMonitor: Моніторинг якості даних
- AlertManager: Менеджер сповіщень
- MonitoringDashboard: Дашборд моніторингу

Використовує:
- Real-time збір метрик
- Threshold-based сповіщення
- Web та text дашборди
- Конфігуруємі інтервали та пороги
"""

from .monitoring_system import (
    MonitoringSystem,
    SystemHealthMonitor,
    ModelPerformanceMonitor,
    DataQualityMonitor,
    AlertManager,
    MonitoringDashboard,
    BaseMonitor,
    AlertSeverity,
    AlertStatus,
    MetricType
)

from .dashboard import (
    MonitoringDashboardGenerator,
    TextBasedDashboard
)

from .config import (
    MonitoringConfig,
    get_monitoring_config,
    create_config_file
)

# Існуючі компоненти для сумісності
from .health_hub import HealthHub
from .infrastructure.resource_monitor import ResourceMonitor

__all__ = [
    # Нові компоненти системи моніторингу
    'MonitoringSystem',
    'SystemHealthMonitor',
    'ModelPerformanceMonitor',
    'DataQualityMonitor',
    'AlertManager',
    'MonitoringDashboard',
    'BaseMonitor',
    'MonitoringDashboardGenerator',
    'TextBasedDashboard',
    'MonitoringConfig',
    'get_monitoring_config',
    'create_config_file',
    'AlertSeverity',
    'AlertStatus',
    'MetricType',

    # Існуючі компоненти
    "HealthHub",
    "ResourceMonitor"
]

__version__ = '1.0.0'