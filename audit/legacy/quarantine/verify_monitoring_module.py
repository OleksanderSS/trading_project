
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.monitoring.infrastructure.resource_monitor import get_resource_monitor
from src.monitoring.monitoring_system import ModelPerformanceMonitor


class TestMonitoringModule(unittest.TestCase):
    def setUp(self):
        try:
            self.resource_monitor = get_resource_monitor()
            self.model_monitor = ModelPerformanceMonitor({})
        except Exception as e:
            self.fail(f"Monitoring module initialization failed: {e}")

    def test_resource_monitor(self):
        """Verify ResourceMonitor is reachable."""
        self.assertIsNotNone(self.resource_monitor)
        # Should return a dict of metrics
        stats = self.resource_monitor.collect_all_metrics()
        self.assertIsInstance(stats, dict)
        print("ResourceMonitor test passed.")

    def test_model_performance_monitor(self):
        """Verify ModelPerformanceMonitor is reachable."""
        self.assertIsNotNone(self.model_monitor)
        print("ModelPerformanceMonitor test passed.")

if __name__ == '__main__':
    unittest.main()
