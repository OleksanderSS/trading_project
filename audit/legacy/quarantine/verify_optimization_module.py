
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.optimization import BayesianOptimizer, PortfolioOptimizer
from src.optimization.factory import OptimizationFactory


class TestOptimizationModule(unittest.TestCase):
    def setUp(self):
        # Initialize factory for testing
        self.factory = OptimizationFactory()

    def test_factory_portfolio(self):
        """Verify factory creates PortfolioOptimizer."""
        optimizer = self.factory.get_optimizer('portfolio', timeframe='1d')
        self.assertIsInstance(optimizer, PortfolioOptimizer)
        print("PortfolioOptimizer factory creation passed.")

    def test_factory_bayesian(self):
        """Verify factory creates BayesianOptimizer."""
        # Bayesian optimizer expects some params, but factory should instantiate it
        optimizer = self.factory.get_optimizer('bayesian')
        self.assertIsInstance(optimizer, BayesianOptimizer)
        print("BayesianOptimizer factory creation passed.")

if __name__ == '__main__':
    unittest.main()
