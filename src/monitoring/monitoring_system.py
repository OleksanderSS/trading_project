"""
Monitoring System - Система моніторингу.

Основні компоненти:
- System Health Monitor: моніторинг здоров'я системи
- Model Performance Monitor: моніторинг продуктивності моделей
- Data Quality Monitor: моніторинг якості даних
- Alert Manager: управління сповіщеннями
- Dashboard Generator: генерація дашбордів

Використовує:
- Real-time metrics collection
- Threshold-based alerting
- Historical performance tracking
- Automated reporting
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import threading
import psutil
import logging
import numpy as np

from src.core.logging.logger import ProjectLogger

class AlertSeverity(Enum):
    """Рівні критичності сповіщень"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Статуси сповіщень"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

class MetricType(Enum):
    """Типи метрик"""
    GAUGE = "gauge"  # Поточне значення
    COUNTER = "counter"  # Кумулятивне значення
    HISTOGRAM = "histogram"  # Розподіл значень
    SUMMARY = "summary"  # Статистичний зведення

class BaseMonitor:
    """Базовий клас для всіх моніторів"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = ProjectLogger.get_logger(f"Monitor.{name}")
        self.metrics = {}
        self.alerts = []
        self.is_running = False

    def start(self):
        """Запуск моніторингу"""
        self.is_running = True
        self.logger.info(f"Started monitoring: {self.name}")

    def stop(self):
        """Зупинка моніторингу"""
        self.is_running = False
        self.logger.info(f"Stopped monitoring: {self.name}")

    def collect_metrics(self) -> Dict[str, Any]:
        """Збір метрик (підкласи повинні перевизначити)"""
        return {}

    def check_thresholds(self):
        """Перевірка порогових значень"""
        pass

    def generate_report(self) -> Dict[str, Any]:
        """Генерація звіту"""
        return {
            'monitor_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'alerts': [alert.__dict__ if hasattr(alert, '__dict__') else alert for alert in self.alerts[-10:]],  # Останні 10
            'status': 'running' if self.is_running else 'stopped'
        }

class SystemHealthMonitor(BaseMonitor):
    """Моніторинг здоров'я системи"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("system_health", config)

        # Пороги
        self.cpu_threshold = self.config.get('cpu_threshold', 80.0)
        self.memory_threshold = self.config.get('memory_threshold', 85.0)
        self.disk_threshold = self.config.get('disk_threshold', 90.0)
        self.network_timeout = self.config.get('network_timeout', 30)

        # Історія метрик
        self.metrics_history = []
        self.history_size = self.config.get('history_size', 100)

    def collect_metrics(self) -> Dict[str, Any]:
        """Збір системних метрик"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_free_gb = disk.free / (1024**3)

            # Network
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv

            # Processes
            process_count = len(psutil.pids())

            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_available_gb': memory_available_gb,
                'disk_percent': disk_percent,
                'disk_used_gb': disk_used_gb,
                'disk_free_gb': disk_free_gb,
                'network_bytes_sent': bytes_sent,
                'network_bytes_recv': bytes_recv,
                'process_count': process_count,
                'timestamp': datetime.now().isoformat()
            }

            # Збереження в історію
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.history_size:
                self.metrics_history.pop(0)

            self.metrics.update(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}

    def check_thresholds(self):
        """Перевірка системних порогових значень"""
        if not self.metrics:
            return

        # CPU check
        if self.metrics.get('cpu_percent', 0) > self.cpu_threshold:
            self._create_alert(
                f"High CPU usage: {self.metrics['cpu_percent']:.1f}%",
                AlertSeverity.WARNING,
                {'cpu_percent': self.metrics['cpu_percent']}
            )

        # Memory check
        if self.metrics.get('memory_percent', 0) > self.memory_threshold:
            self._create_alert(
                f"High memory usage: {self.metrics['memory_percent']:.1f}%",
                AlertSeverity.ERROR,
                {'memory_percent': self.metrics['memory_percent']}
            )

        # Disk check
        if self.metrics.get('disk_percent', 0) > self.disk_threshold:
            self._create_alert(
                f"Low disk space: {self.metrics['disk_percent']:.1f}% used",
                AlertSeverity.CRITICAL,
                {'disk_percent': self.metrics['disk_percent']}
            )

    def _create_alert(self, message: str, severity: AlertSeverity, details: Dict[str, Any]):
        """Створення сповіщення"""
        alert = {
            'id': f"{self.name}_{int(time.time())}",
            'monitor': self.name,
            'message': message,
            'severity': severity.value,
            'status': AlertStatus.ACTIVE.value,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {message}")

class ModelPerformanceMonitor(BaseMonitor):
    """Моніторинг продуктивності моделей"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("model_performance", config)

        # Пороги продуктивності
        self.accuracy_threshold = self.config.get('accuracy_threshold', 0.7)
        self.mae_threshold = self.config.get('mae_threshold', 0.1)
        self.drift_threshold = self.config.get('drift_threshold', 0.05)

        # Модельні метрики
        self.model_metrics = {}
        self.baseline_metrics = {}

    def collect_metrics(self) -> Dict[str, Any]:
        """Збір метрик продуктивності моделей"""
        try:
            # В реальному випадку тут буде інтеграція з системою логування моделей
            # Спрощена версія для демонстрації

            metrics = {
                'total_models': len(self.model_metrics),
                'active_models': sum(1 for m in self.model_metrics.values() if m.get('is_active', True)),
                'models_with_drift': sum(1 for m in self.model_metrics.values() if m.get('drift_detected', False)),
                'average_accuracy': np.mean([m.get('accuracy', 0) for m in self.model_metrics.values()]),
                'timestamp': datetime.now().isoformat()
            }

            self.metrics.update(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting model metrics: {e}")
            return {}

    def update_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Оновлення метрик для моделі"""
        if model_name not in self.baseline_metrics:
            self.baseline_metrics[model_name] = metrics.copy()

        self.model_metrics[model_name] = metrics

        # Перевірка drift
        baseline = self.baseline_metrics[model_name]
        current_accuracy = metrics.get('accuracy', 0)
        baseline_accuracy = baseline.get('accuracy', 0)

        if abs(current_accuracy - baseline_accuracy) > self.drift_threshold:
            self._create_drift_alert(model_name, current_accuracy, baseline_accuracy)

    def _create_drift_alert(self, model_name: str, current_acc: float, baseline_acc: float):
        """Створення сповіщення про drift"""
        alert = {
            'id': f"drift_{model_name}_{int(time.time())}",
            'monitor': self.name,
            'message': f"Model drift detected for {model_name}: accuracy {baseline_acc:.3f} -> {current_acc:.3f}",
            'severity': AlertSeverity.WARNING.value,
            'status': AlertStatus.ACTIVE.value,
            'timestamp': datetime.now().isoformat(),
            'details': {
                'model_name': model_name,
                'baseline_accuracy': baseline_acc,
                'current_accuracy': current_acc,
                'drift': abs(current_acc - baseline_acc)
            }
        }
        self.alerts.append(alert)
        self.logger.warning(f"Drift alert for {model_name}")

class DataQualityMonitor(BaseMonitor):
    """Моніторинг якості даних"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("data_quality", config)

        # Пороги якості
        self.missing_threshold = self.config.get('missing_threshold', 0.05)  # 5%
        self.outlier_threshold = self.config.get('outlier_threshold', 0.1)   # 10%
        self.consistency_threshold = self.config.get('consistency_threshold', 0.95)  # 95%

        # Дані для моніторингу
        self.data_sources = {}
        self.quality_history = {}

    def collect_metrics(self) -> Dict[str, Any]:
        """Збір метрик якості даних"""
        try:
            metrics = {
                'total_sources': len(self.data_sources),
                'sources_with_issues': sum(1 for s in self.data_sources.values() if s.get('has_issues', False)),
                'average_completeness': np.mean([s.get('completeness', 1.0) for s in self.data_sources.values()]),
                'total_missing_values': sum(s.get('missing_count', 0) for s in self.data_sources.values()),
                'timestamp': datetime.now().isoformat()
            }

            self.metrics.update(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting data quality metrics: {e}")
            return {}

    def update_data_quality(self, source_name: str, data_quality_report: Dict[str, Any]):
        """Оновлення звіту якості даних"""
        self.data_sources[source_name] = data_quality_report

        # Перевірка порогових значень
        completeness = data_quality_report.get('completeness', 1.0)
        if completeness < (1 - self.missing_threshold):
            self._create_quality_alert(
                f"Low data completeness in {source_name}: {completeness:.1%}",
                AlertSeverity.WARNING,
                {'source': source_name, 'completeness': completeness}
            )

    def _create_quality_alert(self, message: str, severity: AlertSeverity, details: Dict[str, Any]):
        """Створення сповіщення про якість даних"""
        alert = {
            'id': f"{self.name}_{int(time.time())}",
            'monitor': self.name,
            'message': message,
            'severity': severity.value,
            'status': AlertStatus.ACTIVE.value,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.alerts.append(alert)
        self.logger.warning(f"Data quality alert created: {message}")

class AlertManager:
    """Менеджер сповіщень"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = ProjectLogger.get_logger("AlertManager")

        # Налаштування сповіщень
        self.alert_channels = self.config.get('channels', ['log'])  # log, email, slack, etc.
        self.alert_history = []
        self.active_alerts = {}

        # Автоматичне вирішення
        self.auto_resolve_hours = self.config.get('auto_resolve_hours', 24)

    def process_alert(self, alert: Dict[str, Any]):
        """Обробка нового сповіщення"""
        alert_id = alert['id']

        # Перевірка на дублікати
        if alert_id in self.active_alerts:
            self.logger.debug(f"Duplicate alert ignored: {alert_id}")
            return

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Надсилання сповіщень
        self._send_notifications(alert)

        self.logger.info(f"Alert processed: {alert['message']}")

    def resolve_alert(self, alert_id: str, resolution: str = "auto_resolved"):
        """Вирішення сповіщення"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert['status'] = AlertStatus.RESOLVED.value
            alert['resolved_at'] = datetime.now().isoformat()
            alert['resolution'] = resolution

            del self.active_alerts[alert_id]
            self.logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """Отримання активних сповіщень"""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a['severity'] == severity.value]

        return alerts

    def cleanup_old_alerts(self):
        """Очищення старих сповіщень"""
        cutoff_time = datetime.now() - timedelta(hours=self.auto_resolve_hours)

        to_resolve = []
        for alert_id, alert in self.active_alerts.items():
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time < cutoff_time:
                to_resolve.append(alert_id)

        for alert_id in to_resolve:
            self.resolve_alert(alert_id, "auto_resolved_by_timeout")

    def _send_notifications(self, alert: Dict[str, Any]):
        """Надсилання сповіщень по каналах"""
        for channel in self.alert_channels:
            try:
                if channel == 'log':
                    self._send_log_notification(alert)
                elif channel == 'email':
                    self._send_email_notification(alert)
                elif channel == 'slack':
                    self._send_slack_notification(alert)
                # Додати інші канали за потреби

            except Exception as e:
                self.logger.error(f"Error sending {channel} notification: {e}")

    def _send_log_notification(self, alert: Dict[str, Any]):
        """Надсилання сповіщення в лог"""
        severity = alert['severity'].upper()
        message = f"[{severity}] {alert['message']}"
        if severity == 'ERROR':
            self.logger.error(message)
        elif severity == 'WARNING':
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def _send_email_notification(self, alert: Dict[str, Any]):
        """Надсилання email сповіщення (заглушка)"""
        # Реалізація відправки email
        pass

    def _send_slack_notification(self, alert: Dict[str, Any]):
        """Надсилання Slack сповіщення (заглушка)"""
        # Реалізація відправки в Slack
        pass

class MonitoringDashboard:
    """Генератор дашбордів моніторингу"""

    def __init__(self, monitors: List[BaseMonitor], alert_manager: AlertManager,
                 config: Optional[Dict[str, Any]] = None):
        self.monitors = monitors
        self.alert_manager = alert_manager
        self.config = config or {}
        self.logger = ProjectLogger.get_logger("MonitoringDashboard")

        # Налаштування дашборду
        self.refresh_interval = self.config.get('refresh_interval', 60)  # секунди
        self.history_days = self.config.get('history_days', 7)

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Генерація даних для дашборду"""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'healthy',
                'monitors': {},
                'alerts': {
                    'active': self.alert_manager.get_active_alerts(),
                    'by_severity': self._group_alerts_by_severity(),
                    'recent': self._get_recent_alerts(24)  # останні 24 години
                },
                'summary': self._generate_summary()
            }

            # Збір даних від моніторів
            for monitor in self.monitors:
                monitor_data = monitor.generate_report()
                dashboard_data['monitors'][monitor.name] = monitor_data

                # Перевірка загального статусу
                if monitor_data.get('status') != 'running':
                    dashboard_data['system_status'] = 'degraded'
                if monitor.alerts and any(a.get('severity') in ['error', 'critical'] for a in monitor.alerts[-5:]):
                    dashboard_data['system_status'] = 'unhealthy'

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}

    def _group_alerts_by_severity(self) -> Dict[str, int]:
        """Групування сповіщень по критичності"""
        alerts = self.alert_manager.get_active_alerts()
        grouped = {}

        for severity in AlertSeverity:
            grouped[severity.value] = sum(1 for a in alerts if a['severity'] == severity.value)

        return grouped

    def _get_recent_alerts(self, hours: int) -> List[Dict[str, Any]]:
        """Отримання недавніх сповіщень"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = []

        for alert in self.alert_manager.alert_history:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if alert_time >= cutoff_time:
                recent_alerts.append(alert)

        return recent_alerts[-20:]  # останні 20

    def _generate_summary(self) -> Dict[str, Any]:
        """Генерація зведеного звіту"""
        try:
            total_monitors = len(self.monitors)
            active_monitors = sum(1 for m in self.monitors if m.is_running)
            total_alerts = len(self.alert_manager.get_active_alerts())

            # Статус системи
            if total_alerts > 5:
                system_status = 'critical'
            elif total_alerts > 2:
                system_status = 'warning'
            elif active_monitors < total_monitors:
                system_status = 'degraded'
            else:
                system_status = 'healthy'

            return {
                'system_status': system_status,
                'total_monitors': total_monitors,
                'active_monitors': active_monitors,
                'total_alerts': total_alerts,
                'uptime_percent': (active_monitors / total_monitors * 100) if total_monitors > 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"Error generating summary: {e}")
            return {}

class MonitoringSystem:
    """Головна система моніторингу"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = ProjectLogger.get_logger("MonitoringSystem")

        # Ініціалізація компонентів
        self.system_monitor = SystemHealthMonitor(self.config.get('system_health', {}))
        self.model_monitor = ModelPerformanceMonitor(self.config.get('model_performance', {}))
        self.data_monitor = DataQualityMonitor(self.config.get('data_quality', {}))
        self.alert_manager = AlertManager(self.config.get('alerts', {}))

        self.monitors = [self.system_monitor, self.model_monitor, self.data_monitor]
        self.dashboard = MonitoringDashboard(self.monitors, self.alert_manager, self.config.get('dashboard', {}))

        # Потік моніторингу
        self.monitoring_thread = None
        self.is_running = False
        self.collection_interval = self.config.get('collection_interval', 30)  # секунди

    def start(self):
        """Запуск системи моніторингу"""
        if self.is_running:
            self.logger.warning("Monitoring system is already running")
            return

        self.is_running = True

        # Запуск моніторів
        for monitor in self.monitors:
            monitor.start()

        # Запуск потоку моніторингу
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Monitoring system started")

    def stop(self):
        """Зупинка системи моніторингу"""
        if not self.is_running:
            return

        self.is_running = False

        # Зупинка моніторів
        for monitor in self.monitors:
            monitor.stop()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Monitoring system stopped")

    def _monitoring_loop(self):
        """Основний цикл моніторингу"""
        while self.is_running:
            try:
                # Збір метрик
                for monitor in self.monitors:
                    metrics = monitor.collect_metrics()
                    monitor.check_thresholds()

                    # Обробка сповіщень
                    for alert in monitor.alerts[-5:]:  # останні 5 сповіщень
                        if alert not in [a for a in self.alert_manager.alert_history[-10:]]:  # уникнення дублікатів
                            self.alert_manager.process_alert(alert)

                # Очищення старих сповіщень
                self.alert_manager.cleanup_old_alerts()

                # Затримка
                time.sleep(self.collection_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Коротка затримка перед повтором

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Отримання даних дашборду"""
        return self.dashboard.generate_dashboard_data()

    def get_health_report(self) -> Dict[str, Any]:
        """Отримання звіту про здоров'я системи"""
        return {
            'system_status': 'running' if self.is_running else 'stopped',
            'monitors': {m.name: m.is_running for m in self.monitors},
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'last_collection': datetime.now().isoformat()
        }

    def update_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Оновлення метрик моделі"""
        self.model_monitor.update_model_metrics(model_name, metrics)

    def update_data_quality(self, source_name: str, quality_report: Dict[str, Any]):
        """Оновлення звіту якості даних"""
        self.data_monitor.update_data_quality(source_name, quality_report)