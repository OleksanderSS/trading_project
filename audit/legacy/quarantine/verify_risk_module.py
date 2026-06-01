
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.risk.kill_switch_manager import KillSwitchManager


class TestRiskModule(unittest.TestCase):
    def setUp(self):
        try:
            self.risk_metrics = EliteRiskMetrics()
        except Exception as e:
            self.fail(f"EliteRiskMetrics failed to initialize: {e}")
        
        try:
            self.kill_switch = KillSwitchManager()
        except Exception as e:
            self.fail(f"KillSwitchManager failed to initialize: {e}")

    def test_risk_metrics_init(self):
        """Verify initialization."""
        self.assertIsNotNone(self.risk_metrics)
        print("EliteRiskMetrics initialized.")

    def test_kill_switch_init(self):
        """Verify kill switch initialization."""
        self.assertIsNotNone(self.kill_switch)
        print("KillSwitchManager initialized.")

if __name__ == '__main__':
    unittest.main()
