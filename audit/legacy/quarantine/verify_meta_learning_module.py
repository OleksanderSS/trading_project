
import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.meta_learning.memory.diary_engine import DiaryEngine


class TestMetaLearningModule(unittest.TestCase):
    def setUp(self):
        try:
            self.diary = DiaryEngine()
        except Exception as e:
            self.fail(f"DiaryEngine failed to initialize: {e}")

    def test_diary_init(self):
        """Verify DiaryEngine initialization."""
        self.assertIsNotNone(self.diary)
        print("DiaryEngine initialized.")

    def test_diary_interface(self):
        """Verify core methods exist."""
        self.assertTrue(hasattr(self.diary, 'log_event'))
        print("DiaryEngine interface reachable.")

if __name__ == '__main__':
    unittest.main()
