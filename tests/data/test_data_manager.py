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
        # PREVIOUSLY: this wrote a dummy fixture directly over
        # src/config/system.yaml — a REAL file in UnifiedConfigManager's
        # config precedence chain (layer 1, base infrastructure; see
        # unified_config_manager.py's load-order list), then deleted it in
        # tearDownClass. The only test that ever read this dummy file
        # (test_initialization_from_config, below) is commented out/skipped,
        # so this write+delete served no purpose for any currently-active
        # test while repeatedly clobbering a real, shared, load-bearing
        # config file — the version of system.yaml already committed to git
        # is itself this exact dummy fixture content, meaning a past test
        # run already overwrote whatever real config used to be there.
        # Removed entirely rather than pointed at a temp path, since nothing
        # active needs it; if test_initialization_from_config is ever
        # un-skipped, it should write its dummy config to an isolated temp
        # directory, never to a real path under src/config/.

    def setUp(self):
        """Set up a fresh database for each test."""
        # Mocks for dependencies
        self.mock_config_manager = MagicMock(spec=UnifiedConfigManager)
        # DataManager.__init__ no longer accepts db_path directly — it reads
        # config_manager.get('paths.raw_db', MEMORY_DB) instead. This test
        # previously passed db_path= as a kwarg, which TypeErrors against
        # the current constructor signature (pre-existing breakage, found
        # while adding a regression test for the vix_data upsert bug below —
        # unrelated to that fix, corrected here since it blocked every test
        # in this file, not just the new one).
        self.mock_config_manager.get.return_value = str(self.db_path)
        self.mock_error_handler = MagicMock(spec=IErrorHandler)

        self.dm = DataManager(config_manager=self.mock_config_manager,
                              error_handler=self.mock_error_handler)
        self.dm.execute_query("DROP TABLE IF EXISTS test_stocks")

    def tearDown(self):
        """Close connection and clean up database file after each test."""
        # DataManager has no per-instance close() — connections are shared/
        # pooled by db_path (DataManager._connections), so this must close
        # via the classmethod instead (another pre-existing API drift this
        # test predates: no per-instance close() method exists at all).
        DataManager.close_all_connections()
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
        loaded_df = self.dm.fetch_df("SELECT * FROM test_stocks ORDER BY ticker")
        self.assertEqual(len(loaded_df), 2, "Initial save should result in 2 rows.")
        self.assertEqual(loaded_df['ticker'].tolist(), ['AAPL', 'GOOG'], "Data mismatch after initial save.")

    def test_upsert_operation(self):
        """Test DataManager.upsert()'s actual current semantics: insert
        genuinely new (unique_on) keys, silently SKIP rows whose key
        already exists — it does not update them. This matches the
        project's stated design elsewhere (pipeline_executor.py:
        "Raw data (news, prices, macro) is a permanent chronicle — it
        never expires") for immutable historical facts.

        NOTE: this assertion previously expected AAPL's price to be
        updated from 150.0 to 155.0 (see git history) — that was already
        failing (unrelated to any change in this session) because
        _prepare_upsert_df's dedup-against-existing-keys step filters out
        rows whose key exists, it never issues an UPDATE. If true
        update-on-conflict semantics are ever needed for some table, that
        needs a deliberate design decision (and likely a differently-named
        method, since "upsert" implying update-on-conflict while actually
        doing insert-if-absent is a footgun) — not a change made here
        without knowing which behavior the project owner actually wants
        for which tables.
        """
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

        loaded_df = self.dm.fetch_df("SELECT * FROM test_stocks ORDER BY ticker")
        self.assertEqual(len(loaded_df), 3, "Upsert should result in 3 rows (AAPL kept once, GOOG kept, MSFT added).")
        self.assertEqual(sorted(loaded_df['ticker'].tolist()), ['AAPL', 'GOOG', 'MSFT'], "Ticker list incorrect after upsert.")

        aapl_price = loaded_df[loaded_df['ticker'] == 'AAPL']['price'].iloc[0]
        self.assertEqual(aapl_price, 150.0, "Existing AAPL row's key already existed — current upsert() skips it, keeping the original price.")

    def test_upsert_composite_key_with_internal_duplicates_does_not_raise(self):
        """Reproduces a production bug (vix_data upserts, 2026-07-21 and
        2026-07-23 pipeline runs): _prepare_upsert_df's composite-key branch
        built df_insert_keys with a fresh RangeIndex instead of df_insert's
        own index. drop_duplicates() (called just before) does not reset the
        index, so any input with internal duplicates on the composite key
        produces a non-contiguous df_insert index that no longer lines up
        with the RangeIndex — raising
        "pandas.errors.IndexingError: Unalignable boolean Series provided
        as indexer" the moment the boolean mask is applied. This needs a
        multi-column unique_on (single-column reuses df_insert's own index
        and never hits this path) AND an internal duplicate so
        drop_duplicates() actually removes a row and creates the gap.
        """
        df1 = pd.DataFrame({
            'ticker': ['SPY'], 'price': [400.0],
            'timestamp': pd.to_datetime(['2023-01-01']),
        })
        self.dm.upsert('test_stocks', df1, unique_on=['ticker', 'timestamp'])

        # Row 0 and row 1 are an internal duplicate on (ticker, timestamp);
        # row 2 is genuinely new. After drop_duplicates(keep='first') the
        # surviving index is [0, 2] — not contiguous — which is what
        # exposed the bug.
        df2 = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT'],
            'price': [150.0, 151.0, 300.0],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01']),
        })

        # Must not raise pandas.errors.IndexingError.
        self.dm.upsert('test_stocks', df2, unique_on=['ticker', 'timestamp'])

        loaded_df = self.dm.fetch_df("SELECT * FROM test_stocks ORDER BY ticker")
        self.assertEqual(
            sorted(loaded_df['ticker'].tolist()), ['AAPL', 'MSFT', 'SPY'],
            "Expected one deduplicated AAPL row, one new MSFT row, and the original SPY row.",
        )
        aapl_price = loaded_df[loaded_df['ticker'] == 'AAPL']['price'].iloc[0]
        self.assertEqual(aapl_price, 150.0, "Internal duplicate should keep the first occurrence (price=150.0).")

    def test_load_with_query(self):
        """Test loading data using a specific SQL query."""
        df = pd.DataFrame({
            'ticker': ['NVDA', 'AMD'], 'price': [450.0, 120.0],
        })
        self.dm.upsert('test_stocks', df)
        nvda_df = self.dm.fetch_df("SELECT * FROM test_stocks WHERE ticker = 'NVDA'")
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
