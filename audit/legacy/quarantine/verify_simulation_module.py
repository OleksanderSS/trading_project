
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.simulation.simulation_engine import SimulationEngine


class TestSimulationModule(unittest.TestCase):
    def setUp(self):
        try:
            self.engine = SimulationEngine()
        except Exception as e:
            self.fail(f"SimulationEngine failed to initialize: {e}")

    def test_initialization(self):
        """Verify engine initialization."""
        self.assertIsNotNone(self.engine)
        print("SimulationEngine initialization passed.")

    def test_interface_reachability(self):
        """Verify simulation engine interface is reachable."""
        # Check if basic attributes exist
        self.assertTrue(hasattr(self.engine, 'logger'))
        print("SimulationEngine interface reachable.")

if __name__ == '__main__':
    unittest.main()
