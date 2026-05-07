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

# ✅ NEW: Data Freshness Monitor
from .data_freshness_monitor import (
    DataFreshnessMonitor,
    get_data_freshness_monitor,
    check_freshness_quick
)

# ✅ NEW: Feature Drift Monitor
from .feature_drift_monitor import (
    FeatureDriftMonitor,
    check_feature_drift,
    get_feature_drift_monitor
)

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
    "ResourceMonitor",
    
# ✅ NEW: Data freshness monitoring
    "DataFreshnessMonitor",
    "get_data_freshness_monitor",
    "check_freshness_quick",
    
# ✅ NEW: Feature drift monitoring
    "FeatureDriftMonitor",
    "check_feature_drift",
    "get_feature_drift_monitor"
]

__version__ = '1.0.0'