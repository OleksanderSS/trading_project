
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.backtesting.advanced.advanced_engine import BiasDetector, TransactionCostModel


class TestBacktestingModule(unittest.TestCase):
    def setUp(self):
        self.tc_model = TransactionCostModel()
        self.bias_detector = BiasDetector()

    def test_transaction_cost_model(self):
        """Verify transaction cost calculation logic."""
        costs = self.tc_model.calculate_execution_costs(
            trade_value=100000.0,
            daily_volume=1000000.0,
            volatility=0.02
        )
        self.assertIn('total', costs)
        self.assertGreater(costs['total'], 0)
        self.assertIn('commission', costs)
        print(f"Transaction Cost Test Passed: {costs}")

    def test_bias_detector_instantiation(self):
        """Verify BiasDetector initialization."""
        self.assertEqual(self.bias_detector.lookahead_corr_threshold, 0.5)
        print("Bias Detector Initialization Passed.")

    def test_integration_check(self):
        """Verify backtesting modules are imported correctly."""
        # Check if the modules can be accessed
        try:
            from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer
            optimizer = WalkForwardOptimizer()
            self.assertIsNotNone(optimizer)
            print("WalkForwardOptimizer Instantiation Passed.")
        except Exception as e:
            self.fail(f"Could not initialize WalkForwardOptimizer: {e}")

if __name__ == '__main__':
    unittest.main()
