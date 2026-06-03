"""
Monitoring System Tests - Тести системи моніторингу.

Тестує:
- System monitorинг (CPU, пам'ять, диск)
- Моніторинг моделей
- Моніторинг якості даних
- Alert manager
- Дашборд

Uses:
- unittest для структурованих тестів
- Mock дані для симуляції
- Інтеграційні тести
"""

import unittest
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from src.monitoring.monitoring_system import (
    MonitoringSystem, SystemHealthMonitor, ModelPerformanceMonitor,
    DataQualityMonitor, AlertManager, MonitoringDashboard,
    alertseverity, alertstatus, MetricType
)
from src.monitoring.dashboard import MonitoringDashboardGenerator, TextBasedDashboard
from src.core.logging.logger import ProjectLogger

class TestSystemHealthMonitor(unittest.TestCase):
    """   '"""

    def setUp(self):
        """ """
        self.config = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'disk_threshold': 90.0
        }
        self.monitor = SystemHealthMonitor(self.config)

    @patch('src.monitoring.monitoring_system.psutil')
    def test_collect_metrics(self, mock_psutil):
        """   """
# 
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.virtual_memory.return_value = Mock(
            percent=67.8,
            used=4.2 * (1024**3),
            available=2.1 * (1024**3)
        )
        mock_psutil.disk_usage.return_value = Mock(
            percent=72.3,
            used=150.5 * (1024**3),
            free=58.2 * (1024**3)
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024**3,  # 1 GB
            bytes_recv=512 * (1024**3)  # 0.5 GB
        )
        mock_psutil.pids.return_value = [1, 2, 3, 4, 5]

# 
        metrics = self.monitor.collect_metrics()

# 
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_percent', metrics)
        self.assertIn('disk_percent', metrics)
        self.assertEqual(metrics['cpu_percent'], 45.5)
        self.assertEqual(metrics['memory_percent'], 67.8)
        self.assertEqual(metrics['process_count'], 5)

    @patch('src.monitoring.monitoring_system.psutil')
    def test_threshold_checking(self, mock_psutil):
        """   """
# CPU
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value = Mock(percent=70.0)
        mock_psutil.disk_usage.return_value = Mock(percent=80.0)

        self.monitor.collect_metrics()
        self.monitor.check_thresholds()

# alerts
        self.assertGreater(len(self.monitor.alerts), 0)
        alert = self.monitor.alerts[-1]
        self.assertEqual(alert['severity'], 'warning')
        self.assertIn('High CPU usage', alert['message'])

class TestModelPerformanceMonitor(unittest.TestCase):
    """   """

    def setUp(self):
        self.monitor = ModelPerformanceMonitor({
            'accuracy_threshold': 0.7,
            'drift_threshold': 0.05
        })

    def test_update_model_metrics(self):
        """   """
# 
        self.monitor.update_model_metrics('model_1', {
            'accuracy': 0.85,
            'mae': 0.02,
            'timestamp': datetime.now().isoformat()
        })

# 
        self.assertIn('model_1', self.monitor.baseline_metrics)
        self.assertEqual(self.monitor.baseline_metrics['model_1']['accuracy'], 0.85)

    def test_drift_detection(self):
        """  drift"""
# 
        self.monitor.update_model_metrics('model_1', {'accuracy': 0.85})

# drift
        self.monitor.update_model_metrics('model_1', {'accuracy': 0.75})

# alerts  drift
        self.assertGreater(len(self.monitor.alerts), 0)
        alert = self.monitor.alerts[-1]
        self.assertIn('drift detected', alert['message'].lower())

class TestDataQualityMonitor(unittest.TestCase):
    """   """

    def setUp(self):
        self.monitor = DataQualityMonitor({
            'missing_threshold': 0.05,
            'outlier_threshold': 0.1
        })

    def test_update_data_quality(self):
        """ Update data quality"""
        quality_report = {
            'completeness': 0.92,  # 92% - нижче порогу
            'total_rows': 1000,
            'missing_count': 80,
            'outlier_count': 50
        }

        self.monitor.update_data_quality('source_1', quality_report)

# alerts
        self.assertGreater(len(self.monitor.alerts), 0)
        alert = self.monitor.alerts[-1]
        self.assertIn('Low data completeness', alert['message'])

class TestAlertManager(unittest.TestCase):
    """  """

    def setUp(self):
        self.manager = AlertManager({
            'channels': ['log'],
            'auto_resolve_hours': 24
        })

    def test_process_alert(self):
        """  alerts"""
        alert = {
            'id': 'test_alert_1',
            'monitor': 'test_monitor',
            'message': 'Test alert message',
            'severity': 'warning',
            'status': 'active',
            'timestamp': datetime.now().isoformat(),
            'details': {'test': 'data'}
        }

        self.manager.process_alert(alert)

# 
        self.assertIn('test_alert_1', self.manager.active_alerts)
        self.assertEqual(len(self.manager.alert_history), 1)

    def test_resolve_alert(self):
        """  alerts"""
# alerts
        alert = {
            'id': 'test_alert_2',
            'monitor': 'test_monitor',
            'message': 'Test alert',
            'severity': 'warning',
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        }

        self.manager.process_alert(alert)
        self.assertIn('test_alert_2', self.manager.active_alerts)

# 
        self.manager.resolve_alert('test_alert_2', 'manually resolved')

# 
        self.assertNotIn('test_alert_2', self.manager.active_alerts)
        self.assertEqual(self.manager.active_alerts['test_alert_2']['status'], 'resolved')

    def test_get_active_alerts_by_severity(self):
        """    """
        alerts = [
            {
                'id': 'alert_1',
                'severity': 'warning',
                'monitor': 'test',
                'message': 'Warning alert',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'alert_2',
                'severity': 'error',
                'monitor': 'test',
                'message': 'Error alert',
                'timestamp': datetime.now().isoformat()
            }
        ]

        for alert in alerts:
            self.manager.process_alert(alert)

# 
        warnings = self.manager.get_active_alerts(alertseverity.WARNING)
        errors = self.manager.get_active_alerts(alertseverity.ERROR)

        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(errors), 1)

class TestMonitoringDashboard(unittest.TestCase):
    """  """

    def setUp(self):
# 
        self.mock_monitor1 = Mock()
        self.mock_monitor1.name = 'system_health'
        self.mock_monitor1.is_running = True
        self.mock_monitor1.alerts = []
        self.mock_monitor1.generate_report.return_value = {
            'monitor_name': 'system_health',
            'status': 'running',
            'metrics': {'cpu_percent': 45.0},
            'alerts': []
        }

        self.mock_monitor2 = Mock()
        self.mock_monitor2.name = 'model_performance'
        self.mock_monitor2.is_running = True
        self.mock_monitor2.alerts = []
        self.mock_monitor2.generate_report.return_value = {
            'monitor_name': 'model_performance',
            'status': 'running',
            'metrics': {'total_models': 3},
            'alerts': []
        }

# Alert manager
        self.mock_alert_manager = Mock()
        self.mock_alert_manager.get_active_alerts.return_value = []
        self.mock_alert_manager.alert_history = []

        self.dashboard = MonitoringDashboard(
            [self.mock_monitor1, self.mock_monitor2],
            self.mock_alert_manager
        )

    def test_generate_dashboard_data(self):
        """   """
        data = self.dashboard.generate_dashboard_data()

# 
        self.assertIn('system_status', data)
        self.assertIn('monitors', data)
        self.assertIn('alerts', data)
        self.assertIn('summary', data)

# 
        self.assertIn('system_health', data['monitors'])
        self.assertIn('model_performance', data['monitors'])

# 
        summary = data['summary']
        self.assertIn('total_monitors', summary)
        self.assertIn('active_monitors', summary)
        self.assertEqual(summary['total_monitors'], 2)
        self.assertEqual(summary['active_monitors'], 2)

class TestTextBasedDashboard(unittest.TestCase):
    """  """

    def setUp(self):
# 
        self.mock_monitoring_system = Mock()
        self.mock_monitoring_system.get_dashboard_data.return_value = {
            'system_status': 'healthy',
            'summary': {
                'active_monitors': 3,
                'total_monitors': 3,
                'total_alerts': 1
            },
            'monitors': {
                'system_health': {
                    'metrics': {
                        'cpu_percent': 45.5,
                        'memory_percent': 67.8,
                        'disk_percent': 72.3
                    }
                },
                'model_performance': {
                    'metrics': {
                        'total_models': 5,
                        'active_models': 4,
                        'average_accuracy': 0.82
                    }
                }
            },
            'alerts': {
                'active': [{
                    'severity': 'warning',
                    'monitor': 'system_health',
                    'message': 'High CPU usage',
                    'timestamp': datetime.now().isoformat()
                }],
                'recent': []
            }
        }

        self.dashboard = TextBasedDashboard(self.mock_monitoring_system)

    def test_generate_report(self):
        """   """
        report = self.dashboard.generate_report()

# 
        self.assertIn('TRADING SYSTEM MONITORING DASHBOARD', report)
        self.assertIn('System Status: HEALTHY', report)
        self.assertIn('SYSTEM METRICS:', report)
        self.assertIn('MODEL PERFORMANCE:', report)
        self.assertIn('ACTIVE alerts:', report)

# 
        self.assertIn('CPU Usage: 45.5%', report)
        self.assertIn('Memory Usage: 67.8%', report)
        self.assertIn('Total Models: 5', report)

    def test_save_report(self):
        """ Save report"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp_path = tmp.name

        try:
            self.dashboard.save_report(tmp_path)

# 
            self.assertTrue(os.path.exists(tmp_path))

# 
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('TRADING SYSTEM MONITORING DASHBOARD', content)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestMonitoringSystem(unittest.TestCase):
    """   """

    def setUp(self):
        self.config = {
            'collection_interval': 1,  # швидкий інтервал для тестів
            'system_health': {'cpu_threshold': 90.0},
            'model_performance': {'accuracy_threshold': 0.8},
            'data_quality': {'missing_threshold': 0.1},
            'alerts': {'channels': ['log']}
        }
        self.system = MonitoringSystem(self.config)

    def tearDown(self):
        """Cleanup  """
        if self.system.is_running:
            self.system.stop()

    def test_initialization(self):
        """  """
        self.assertIsInstance(self.system.system_monitor, SystemHealthMonitor)
        self.assertIsInstance(self.system.model_monitor, ModelPerformanceMonitor)
        self.assertIsInstance(self.system.data_monitor, DataQualityMonitor)
        self.assertIsInstance(self.system.alert_manager, AlertManager)
        self.assertIsInstance(self.system.dashboard, MonitoringDashboard)

        self.assertFalse(self.system.is_running)

    def test_start_stop(self):
        """    """
# 
        self.system.start()
        self.assertTrue(self.system.is_running)

# 
        time.sleep(0.1)

# 
        self.system.stop()
        self.assertFalse(self.system.is_running)

    def test_get_health_report(self):
        """    '"""
        report = self.system.get_health_report()

        expected_keys = ['system_status', 'monitors', 'active_alerts', 'last_collection']
        for key in expected_keys:
            self.assertIn(key, report)

        self.assertEqual(report['system_status'], 'stopped')  # Система не запущена

    def test_update_model_metrics(self):
        """     """
        metrics = {'accuracy': 0.85, 'mae': 0.03}

        self.system.update_model_metrics('test_model', metrics)

# 
        self.assertIn('test_model', self.system.model_monitor.model_metrics)

    def test_update_data_quality(self):
        """ Update data quality  """
        quality_report = {
            'completeness': 0.95,
            'total_rows': 1000,
            'missing_count': 50
        }

        self.system.update_data_quality('test_source', quality_report)

# 
        self.assertIn('test_source', self.system.data_monitor.data_sources)

class TestMonitoringDashboardGenerator(unittest.TestCase):
    """  """

    def setUp(self):
        self.mock_system = Mock()
        self.mock_system.get_dashboard_data.return_value = {
            'system_status': 'healthy',
            'summary': {'active_monitors': 2, 'total_monitors': 2},
            'alerts': {'active': []}
        }

        self.config = {
            'auto_save': False,
            'save_path': 'test_reports'
        }

        self.generator = MonitoringDashboardGenerator(self.mock_system, self.config)

    def test_initialization(self):
        """  """
        self.assertIsInstance(self.generator.text_dashboard, TextBasedDashboard)

# Web dashboard   None  Plotly
        if 'plotly' in globals():
            self.assertIsNotNone(self.generator.web_dashboard)
        else:
            self.assertIsNone(self.generator.web_dashboard)

    def test_generate_text_report(self):
        """   """
        report = self.generator.generate_text_report()

        self.assertIsInstance(report, str)
        self.assertIn('TRADING SYSTEM MONITORING DASHBOARD', report)

    def test_save_current_report(self):
        """   """
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, 'test_report.txt')

            self.generator.save_current_report(filepath)

# 
            self.assertTrue(os.path.exists(filepath))

    def test_get_dashboard_summary(self):
        """   """
        summary = self.generator.get_dashboard_summary()

        expected_keys = ['system_status', 'active_monitors', 'total_alerts', 'last_update']
        for key in expected_keys:
            self.assertIn(key, summary)

        self.assertEqual(summary['system_status'], 'healthy')
        self.assertEqual(summary['active_monitors'], 2)

if __name__ == '__main__':
# Configure logging
    logging.basicConfig(level=logging.WARNING)

# 
    unittest.main(verbosity=2)
