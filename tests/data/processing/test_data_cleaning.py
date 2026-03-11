#!/usr/bin/env python3
"""
Test suite for data cleaning utilities.
"""

import os
import sys
import pandas as pd
import numpy as np
import unittest
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the functions to be tested
from src.data.processing.data_cleaning import harmonize_dataframe, safe_fill

class TestDataCleaning(unittest.TestCase):
    """Test cases for data cleaning functions."""

    def test_harmonize_dataframe(self):
        """Test the harmonize_dataframe function."""
        # 1. Test with duplicate columns
        df_dupe_cols = pd.DataFrame(np.ones((5, 4)), columns=['A', 'B', 'A', 'C'])
        harmonized_df = harmonize_dataframe(df_dupe_cols)
        self.assertEqual(list(harmonized_df.columns), ['A', 'B', 'C'], "Duplicate columns were not removed correctly.")

        # 2. Test with object and mixed types
        df_mixed_types = pd.DataFrame({
            'floats': [1.0, 2.0, np.nan, 4.0],
            'objects': ['a', 'b', 'c', 'nan'],
            'ints': [1, 2, 3, 4]
        })
        harmonized_df = harmonize_dataframe(df_mixed_types)
        self.assertEqual(harmonized_df['objects'].dtype, 'string', "Object column not converted to string.")
        self.assertEqual(harmonized_df['objects'].iloc[3], '', "'nan' string should be replaced with empty string.")
        
        # 3. Test with fully empty columns
        df_empty_cols = pd.DataFrame({'A': [1, 2, 3], 'B': [np.nan, np.nan, np.nan]})
        harmonized_df = harmonize_dataframe(df_empty_cols, dropna_cols=True)
        self.assertNotIn('B', harmonized_df.columns, "Empty columns were not dropped.")

    def test_safe_fill(self):
        """Test the safe_fill function."""
        # Create a mock config for testing safe_fill
        mock_config = {
            'safe_fill': {
                'fill_with_zero': ['sentiment']
            }
        }

        df_with_nans = pd.DataFrame({
            'numeric': [1.0, np.nan, 3.0, np.nan],
            'sentiment': [0.5, np.nan, -0.5, 0.0],
            'category': ['X', 'Y', np.nan, 'Z']
        })

        # Can't test with the real config loader, so we check the logic
        # by manually calling the parts or assuming default behavior
        filled_df = safe_fill(df_with_nans)

        # Test numeric fill (ffill + bfill)
        self.assertFalse(filled_df['numeric'].isna().any(), "Numeric column should be filled.")
        self.assertEqual(filled_df['numeric'].tolist(), [1.0, 1.0, 3.0, 3.0], "Numeric fill logic is incorrect.")

        # Test categorical fill
        self.assertFalse(filled_df['category'].isna().any(), "Category column should be filled.")
        self.assertEqual(filled_df['category'].iloc[2], "unknown", "Categorical NaNs should be filled with 'unknown'.")

if __name__ == "__main__":
    unittest.main()
