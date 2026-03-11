#!/usr/bin/env python3
"""
Test suite for the DataManager class.
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Ensure the project root is in the Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Now, we can import the modules
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import IErrorHandler

class TestDataManager(unittest.TestCase):
    """Test cases for the DataManager."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        cls.db_path = project_root / "data" / "test_main.duckdb"
        cls.dummy_config_path = project_root / "src"/ "config" / "system.yaml"

        # Create a dummy config for testing config-based initialization
        with open(cls.dummy_config_path, "w") as f:
            f.write(f"storage:\n  db_path: {cls.db_path.as_posix()}\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment after all tests."""
        if os.path.exists(cls.dummy_config_path):
            os.remove(cls.dummy_config_path)

    def setUp(self):
        """Set up a fresh database for each test."""
        # Mocks for dependencies
        self.mock_config_manager = MagicMock(spec=UnifiedConfigManager)
        self.mock_error_handler = MagicMock(spec=IErrorHandler)

        self.dm = DataManager(config_manager=self.mock_config_manager, 
                              error_handler=self.mock_error_handler, 
                              db_path=str(self.db_path))
        self.dm.execute_query("DROP TABLE IF EXISTS test_stocks")

    def tearDown(self):
        """Close connection and clean up database file after each test."""
        self.dm.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        wal_file = Path(f"{self.db_path}.wal")
        if wal_file.exists():
            wal_file.unlink()

    def test_initial_save(self):
        """Test the initial saving of a DataFrame."""
        df1 = pd.DataFrame({
            'ticker': ['AAPL', 'GOOG'], 'price': [150.0, 2800.0],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01'])
        })
        self.dm.upsert('test_stocks', df1)
        loaded_df = self.dm.load_data("SELECT * FROM test_stocks ORDER BY ticker")
        self.assertEqual(len(loaded_df), 2, "Initial save should result in 2 rows.")
        self.assertEqual(loaded_df['ticker'].tolist(), ['AAPL', 'GOOG'], "Data mismatch after initial save.")

    def test_upsert_operation(self):
        """Test the upsert functionality (update existing, insert new)."""
        df1 = pd.DataFrame({
            'ticker': ['AAPL', 'GOOG'], 'price': [150.0, 2800.0],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01'])
        })
        self.dm.upsert('test_stocks', df1)

        df2 = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'], 'price': [155.0, 300.0],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01'])
        })
        self.dm.upsert('test_stocks', df2, unique_on=['ticker', 'timestamp'])

        loaded_df = self.dm.load_data("SELECT * FROM test_stocks ORDER BY ticker")
        self.assertEqual(len(loaded_df), 3, "Upsert should result in 3 rows.")
        self.assertEqual(sorted(loaded_df['ticker'].tolist()), ['AAPL', 'GOOG', 'MSFT'], "Ticker list incorrect after upsert.")
        
        aapl_price = loaded_df[loaded_df['ticker'] == 'AAPL']['price'].iloc[0]
        self.assertEqual(aapl_price, 155.0, "AAPL price should have been updated.")

    def test_load_with_query(self):
        """Test loading data using a specific SQL query."""
        df = pd.DataFrame({
            'ticker': ['NVDA', 'AMD'], 'price': [450.0, 120.0],
        })
        self.dm.upsert('test_stocks', df)
        nvda_df = self.dm.load_data("SELECT * FROM test_stocks WHERE ticker = 'NVDA'")
        self.assertEqual(len(nvda_df), 1, "Query should return a single row.")
        self.assertEqual(nvda_df['price'].iloc[0], 450.0, "Incorrect data retrieved by query.")

    # @unittest.skip("Skipping until config handling in tests is refactored.")
    # def test_initialization_from_config(self):
    #     """Test that the DataManager can initialize using a configuration file."""
    #     # Close the instance created in setUp
    #     self.dm.close()
        
    #     # This should now read from the dummy config file
    #     dm_config = DataManager()
    #     self.assertTrue(str(self.db_path) in dm_config.db_path, "DataManager did not load the path from config.")
    #     # The file is cleaned up in tearDown

if __name__ == "__main__":
    unittest.main()
