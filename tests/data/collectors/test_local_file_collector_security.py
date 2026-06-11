import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Any
from unittest.mock import MagicMock
from src.data.collectors.local_file_collector import LocalFileCollector

# Concrete class for testing
class TestableLocalFileCollector(LocalFileCollector):
    async def run(self, tickers: List[str], **kwargs) -> Optional[Any]:
        return None

class TestLocalFileCollectorSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.test_dir.name).resolve()
        
        # Create a valid file within base_dir
        self.valid_file = self.base_dir / "data.csv"
        df = pd.DataFrame({"col1": [1, 2]})
        df.to_csv(self.valid_file, index=False)

        # Mock dependencies
        self.mock_factory = MagicMock()
        self.mock_db = MagicMock()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_invalid_path_blocked(self):
        # Attempt to configure with a path outside the project
        invalid_path = "../../outside.csv"
        configs = {"file_path": invalid_path, "file_type": "csv"}
        
        # Collector should log error and set file_path to None
        collector = TestableLocalFileCollector(configs, self.mock_factory, self.mock_db)
        self.assertIsNone(collector.file_path)

if __name__ == '__main__':
    unittest.main()
