
import os
import sys
import unittest

# Add project root
sys.path.append(os.getcwd())

from src.trading.portfolio_manager import PortfolioManager
from src.trading.trader import Trader
from src.trading.trading_orchestrator import TradingOrchestrator
from src.trading.virtual_portfolio import VirtualPortfolio


# Mocking dependencies for verification
class MockConsensusEngine: pass
class MockPostInferenceFilter: pass

class TestTradingModule(unittest.TestCase):
    def setUp(self):
        # Instantiate core components to verify wiring
        self.portfolio = VirtualPortfolio(initial_balance=100000.0)
        self.portfolio_manager = PortfolioManager(virtual_portfolio=self.portfolio)
        self.trader = Trader()
        
        self.orchestrator = TradingOrchestrator(
            consensus_engine=MockConsensusEngine(),
            portfolio_manager=self.portfolio_manager,
            virtual_portfolio=self.portfolio,
            trader=self.trader,
            post_inference_filter=MockPostInferenceFilter()
        )

    def test_orchestrator_initialization(self):
        """Verify orchestrator successfully initializes with components."""
        self.assertIsNotNone(self.orchestrator.portfolio_manager)
        self.assertIsNotNone(self.orchestrator.trader)
        print("Orchestrator initialization passed.")

    def test_portfolio_manager(self):
        """Verify PortfolioManager base logic."""
        self.assertEqual(self.portfolio.current_balance, 100000.0)
        print("Portfolio initialization passed.")

if __name__ == '__main__':
    unittest.main()
