
import os
import sys
import unittest

# Add project root
sys.path.append(os.getcwd())

from src.training.unified_training_manager import TrainingStrategy, UnifiedTrainingManager


class TestTrainingModule(unittest.TestCase):
    def setUp(self):
        # Prevent actual training dir creation by mocking or using a temp path
        # Assuming UnifiedTrainingManager handles path existence gracefully.
        try:
            self.manager = UnifiedTrainingManager()
        except Exception as e:
            self.fail(f"UnifiedTrainingManager initialization failed: {e}")

    def test_initialization(self):
        """Verify trainer initialization."""
        self.assertIn(TrainingStrategy.BATCH.value, self.manager.trainers)
        self.assertIn(TrainingStrategy.PROGRESSIVE.value, self.manager.trainers)
        print("Initialization test passed.")

    def test_strategy_selection(self):
        """Verify strategy recommendation logic."""
        tickers_small = ['AAPL', 'MSFT', 'GOOG', 'AMD', 'NVDA']
        analysis_small = self.manager.analyze_ticker_set(tickers_small)
        self.assertEqual(analysis_small['recommended_strategy'], 'batch')
        
        tickers_large = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        analysis_large = self.manager.analyze_ticker_set(tickers_large)
        self.assertEqual(analysis_large['recommended_strategy'], 'progressive')
        print("Strategy selection logic passed.")

    def test_plan_creation(self):
        """Verify plan structure."""
        tickers = ['AAPL', 'MSFT']
        plan = self.manager.create_unified_plan(tickers)
        self.assertIn('strategy', plan)
        self.assertIn('tickers', plan)
        self.assertIn('ticker_plans', plan)
        print("Plan creation logic passed.")

if __name__ == '__main__':
    unittest.main()
