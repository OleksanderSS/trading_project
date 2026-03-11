#!/usr/bin/env python3
"""
Test suite for the DataValidator class.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import unittest

# Ensure the project root is in the Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Now, we can import the module
from src.data.data_validator import DataValidator, OHLCVRowValidator

class TestDataValidator(unittest.TestCase):
    """Test cases for the DataValidator."""

    @classmethod
    def setUpClass(cls):
        """Set up basic data structures for all tests."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Base index with a duplicate for testing time-series integrity
        base_index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03'])

        # A pristine, valid DataFrame structure (before deduplication)
        cls.df_valid = pd.DataFrame({
            'open': [100.0, 102.0, 103.0, 104.0],
            'high': [103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 101.0, 102.0, 103.0],
            'close': [102.0, 103.0, 104.0, 105.0],
            'volume': [10000, 12000, 11000, 13000]
        }, index=base_index)

    def test_valid_dataframe_with_duplicates(self):
        """Test a DataFrame that is valid but contains duplicate timestamps."""
        is_ok, msgs = DataValidator.validate_ohlcv(self.df_valid)
        self.assertFalse(is_ok, "Validation should fail due to duplicate index.")
        self.assertTrue(any("Duplicate timestamps" in m for m in msgs), "Missing duplicate timestamp warning.")

    def test_perfect_dataframe(self):
        """Test a perfectly valid DataFrame (no duplicates)."""
        df_no_dupes = self.df_valid.loc[~self.df_valid.index.duplicated()]
        is_ok, msgs = DataValidator.validate_ohlcv(df_no_dupes, use_pydantic=True)
        self.assertTrue(is_ok, f"Validation failed for a good DataFrame. Messages: {msgs}")

    def test_missing_columns(self):
        """Test DataFrame with missing required columns."""
        df_missing_col = self.df_valid.drop(columns=['volume'])
        is_ok, msgs = DataValidator.validate_ohlcv(df_missing_col)
        self.assertFalse(is_ok, "Validation should fail for missing columns.")
        self.assertTrue(any("Missing required columns" in m for m in msgs), "Incorrect error for missing columns.")

    def test_null_values(self):
        """Test DataFrame with null values."""
        df_null = self.df_valid.copy()
        df_null.loc['2023-01-01', 'close'] = np.nan
        is_ok, msgs = DataValidator.validate_ohlcv(df_null)
        self.assertFalse(is_ok, "Validation should fail for null values.")
        self.assertTrue(any("Null values found" in m for m in msgs), "Incorrect error for null values.")

    def test_bad_logic(self):
        """Test DataFrame with logical inconsistencies (e.g., high < low)."""
        df_bad_logic = self.df_valid.copy()
        df_bad_logic.loc['2023-01-01', 'high'] = 98.0  # high < low
        is_ok, msgs = DataValidator.validate_ohlcv(df_bad_logic, use_pydantic=True)
        self.assertFalse(is_ok, "Validation should fail for bad logic.")
        self.assertTrue(any("high < low" in m for m in msgs) or any("row-level validation failed" in m for m in msgs), "Incorrect error for bad logic.")

    def test_negative_price(self):
        """Test DataFrame with negative price values."""
        df_negative = self.df_valid.copy()
        df_negative.loc['2023-01-01', 'low'] = -99.0
        is_ok, msgs = DataValidator.validate_ohlcv(df_negative)
        self.assertFalse(is_ok, "Validation should fail for negative prices.")
        self.assertTrue(any("non-positive price" in m for m in msgs), "Incorrect error for negative price.")

    def test_bad_data_type(self):
        """Test DataFrame with incorrect data types."""
        df_bad_type = self.df_valid.copy()
        df_bad_type['open'] = df_bad_type['open'].astype(str)
        is_ok, msgs = DataValidator.validate_ohlcv(df_bad_type)
        self.assertFalse(is_ok, "Validation should fail for wrong data types.")
        self.assertTrue(any("not numeric" in m for m in msgs), "Incorrect error for bad data type.")

    def test_no_datetime_index(self):
        """Test DataFrame without a DatetimeIndex."""
        df_no_dt_index = self.df_valid.reset_index()
        is_ok, msgs = DataValidator.validate_ohlcv(df_no_dt_index)
        self.assertFalse(is_ok, "Validation should fail for non-DatetimeIndex.")
        self.assertTrue(any("must be a DatetimeIndex" in m for m in msgs), "Incorrect error for missing DatetimeIndex.")

if __name__ == "__main__":
    unittest.main()
