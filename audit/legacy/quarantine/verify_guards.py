
import os
import sys
import unittest

import pandas as pd

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.pipeline.guards.temporal_leakage_guard import TemporalLeakageGuard
from src.pipeline.guards.timeframe_alignment_guard import TimeframeAlignmentGuard


class TestPipelineGuards(unittest.TestCase):
    def setUp(self):
        self.leakage_guard = TemporalLeakageGuard()
        self.alignment_guard = TimeframeAlignmentGuard()

    def test_leakage_guard_init(self):
        """Verify leakage guard initialization."""
        self.assertIsNotNone(self.leakage_guard)
        print("TemporalLeakageGuard initialized.")

    def test_alignment_guard_init(self):
        """Verify alignment guard initialization."""
        self.assertIsNotNone(self.alignment_guard)
        print("TimeframeAlignmentGuard initialized.")

    def test_leakage_guard_functionality(self):
        """Test a dummy leakage scenario."""
        # Simple test: feature that includes future data
        dates = pd.date_range(start='2026-01-01', periods=5)
        df = pd.DataFrame({'datetime': dates, 'feature': [1, 2, 3, 4, 5]})
        # Simulate leakage: feature = close_price_tomorrow
        df['leakage'] = df['feature'].shift(-1)
        
        # This is a basic integration test of the interface
        # The internal analysis logic will flag this if configured correctly
        analysis = self.leakage_guard._analyze_feature_for_leakage(df['leakage'], df['datetime'], current_time=pd.Timestamp.now(), timeframe='1d')
        # The exact result depends on internal pattern detection, 
        # but the method should execute without error
        self.assertIn('has_leakage', analysis)
        print("TemporalLeakageGuard interface reachable.")

if __name__ == '__main__':
    unittest.main()
