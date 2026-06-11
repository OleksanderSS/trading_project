
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.models.enhanced_factory import EnhancedModelFactory
from src.models.integrated_model_manager import get_integrated_model_manager
from src.models.model_pool import get_model_pool


class TestModelsModule(unittest.TestCase):
    def setUp(self):
        try:
            self.manager = get_integrated_model_manager()
            self.pool = get_model_pool()
            self.factory = EnhancedModelFactory()
        except Exception as e:
            self.fail(f"Models module initialization failed: {e}")

    def test_manager(self):
        """Verify IntegratedModelManager initialization."""
        self.assertIsNotNone(self.manager)
        print("IntegratedModelManager initialized.")

    def test_pool(self):
        """Verify ModelPool initialization."""
        self.assertIsNotNone(self.pool)
        print("ModelPool initialized.")

    def test_factory(self):
        """Verify EnhancedModelFactory initialization."""
        self.assertIsNotNone(self.factory)
        print("EnhancedModelFactory initialized.")

if __name__ == '__main__':
    unittest.main()
