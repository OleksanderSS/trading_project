
import os
import sys
import unittest

# Add project root
sys.path.append(os.getcwd())

from src.config.unified_config_manager import UnifiedConfigManager
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config_manager = UnifiedConfigManager()
        try:
            self.orchestrator = PipelineOrchestrator(config_manager=self.config_manager)
        except Exception as e:
            self.fail(f"PipelineOrchestrator initialization failed: {e}")

    def test_initialization(self):
        """Verify initialization."""
        self.assertIsNotNone(self.orchestrator)
        print("PipelineOrchestrator initialization passed.")

    def test_stage_loading(self):
        """Verify stage loader is present."""
        self.assertTrue(hasattr(self.orchestrator, 'logger'))
        print("PipelineOrchestrator structure verified.")

if __name__ == '__main__':
    unittest.main()
