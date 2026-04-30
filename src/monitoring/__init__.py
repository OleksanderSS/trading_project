"""
Monitoring Package - Пакет системи моніторингу.

Містить:
- MonitoringSystem: Головна система моніторингу
- SystemHealthMonitor: Моніторинг системного здоров'я
- ModelPerformanceMonitor: Моніторинг продуктивності моделей
- DataQualityMonitor: Моніторинг якості даних
- AlertManager: Alert manager
- MonitoringDashboard: Monitoring Dashboard

Uses:
- Real-time збір метрик
- Threshold-based alerts
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
    alertseverity,
    alertstatus,
    MetricType
)

try:
    from .dashboard import (
        MonitoringDashboardGenerator,
        TextBasedDashboard
    )
except ImportError:
    MonitoringDashboardGenerator = None
    TextBasedDashboard = None

from .config import (
    MonitoringConfig,
    get_monitoring_config,
    create_config_file
)

# 
from .health_hub import HealthHub
from .infrastructure.resource_monitor import ResourceMonitor

__all__ = [
# 
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
    'alertseverity',
    'alertstatus',
    'MetricType',

# 
    "HealthHub",
    "ResourceMonitor"
]

__version__ = '1.0.0'