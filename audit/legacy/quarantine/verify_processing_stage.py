
import os
import sys
import unittest

import pandas as pd

# Add project root
sys.path.append(os.getcwd())

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.pipeline.stages.stage_2_processing import ProcessingStage


class TestProcessingStage(unittest.TestCase):
    def setUp(self):
        # Setup minimal dependencies
        self.config_manager = UnifiedConfigManager()
        self.error_handler = ErrorHandler()
        
        try:
            self.stage = ProcessingStage(self.config_manager, self.error_handler)
        except Exception as e:
            self.fail(f"ProcessingStage initialization failed: {e}")

    def test_initialization(self):
        """Verify initialization."""
        self.assertIsNotNone(self.stage)
        self.assertIsNotNone(self.stage.data_filter)
        self.assertIsNotNone(self.stage.normalization_manager)
        print("ProcessingStage initialization passed.")

    def test_price_preprocessing(self):
        """Verify PricePreprocessor is reachable and basic logic works."""
        from src.processing.price_preprocessor import PricePreprocessor
        preprocessor = PricePreprocessor()
        
        df = pd.DataFrame({'close': [100, 101, 102]})
        # Expected behavior: ensure columns are lowercase/standard
        processed = preprocessor.normalize_price_df(df)
        self.assertIn('close', processed.columns)
        print("PricePreprocessor logic passed.")

if __name__ == '__main__':
    unittest.main()
