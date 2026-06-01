
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.config.unified_config_manager import UnifiedConfigManager
from src.main.system_orchestrator import SystemOrchestrator


class TestMainModule(unittest.TestCase):
    def setUp(self):
        self.config_manager = UnifiedConfigManager()
        self.orchestrator = SystemOrchestrator(config_manager=self.config_manager)

    def test_orchestrator_init(self):
        """Verify SystemOrchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        print("SystemOrchestrator initialized.")

    def test_run_mode_interface(self):
        """Verify the interface for running modes exists."""
        # Check if the run_mode method exists (async)
        self.assertTrue(hasattr(self.orchestrator, 'run_mode'))
        print("SystemOrchestrator interface is valid.")

if __name__ == '__main__':
    unittest.main()
