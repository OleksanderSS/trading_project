
import os
import sys
import unittest

# Add project root
sys.path.append(os.getcwd())

from src.patterns.pattern_analyzer import PatternAnalyzer


class TestPatternsModule(unittest.TestCase):
    def setUp(self):
        # The constructor needs access to config or minimal setup
        # Passing a dummy config object or None as it tries to access config
        self.analyzer = PatternAnalyzer(enable_debug=True)

    def test_initialization(self):
        """Verify PatternAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        print("PatternAnalyzer initialized.")

    def test_pattern_detection_interface(self):
        """Verify analyzer interface."""
        # Simple test to see if methods exist
        self.assertTrue(hasattr(self.analyzer, 'analyze'))
        print("PatternAnalyzer interface reachable.")

if __name__ == '__main__':
    unittest.main()
