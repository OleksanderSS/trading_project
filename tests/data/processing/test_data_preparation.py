#!/usr/bin/env python3
"""
Test suite for the DataPreparer class.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Mock UnifiedConfigManager before importing DataPreparer
from unittest.mock import MagicMock

# Now import the class to be tested
from src.data.processing.data_preparation import DataPreparer

# Create a mock config manager
mock_config = {
    'data.processing.preparation': {
        'test_size': 0.2,
        'seq_len': 5,
        'feature_columns': ['feature1', 'feature2'],
        'targets_config': [
            {
                'name': 'target_reg',
                'type': 'regression',
                'params': {'base_col': 'close', 'shift': -1}
            },
            {
                'name': 'target_bin',
                'type': 'classification_binary',
                'params': {'base_col': 'close', 'shift': -1, 'threshold': 0.001}
            }
        ]
    }
}

mock_config_manager = MagicMock()
mock_config_manager.get_specific_config.side_effect = lambda *keys: mock_config.get('.'.join(keys))

class TestDataPreparer(unittest.TestCase):
    """Test cases for the DataPreparer."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        cls.preparer = DataPreparer(config_manager=mock_config_manager)
        
        # Create a sample DataFrame
        date_rng = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        cls.df = pd.DataFrame(date_rng, columns=['date'])
        cls.df['close'] = np.random.uniform(100, 200, size=(len(date_rng)))
        cls.df['feature1'] = np.random.rand(len(date_rng))
        cls.df['feature2'] = np.random.rand(len(date_rng))
        cls.df = cls.df.set_index('date')

    def test_target_generation(self):
        """Test if target variables are generated correctly."""
        df_with_targets = self.preparer.target_generator.generate_targets(self.df)
        self.assertIn('target_reg', df_with_targets.columns)
        self.assertIn('target_bin', df_with_targets.columns)
        # Check for NaNs created by shifting
        self.assertTrue(df_with_targets['target_reg'].isna().any())

    def test_prepare_data_for_models(self):
        """Test the main data preparation pipeline."""
        data_packages = self.preparer.prepare_data_for_models(self.df)
        
        # Check root structure
        self.assertIn('light_models', data_packages)
        self.assertIn('heavy_models', data_packages)

        # --- Test Light Models Package ---
        light_data = data_packages['light_models']
        self.assertIn('X_train', light_data)
        self.assertIn('y_test', light_data)
        self.assertEqual(light_data['X_train'].shape[1], 2) # Two feature columns
        self.assertIn('target_reg', light_data['y_train'])

        # --- Test Heavy Models Package ---
        heavy_data = data_packages['heavy_models']
        seq_len = mock_config['data.processing.preparation']['seq_len']
        
        self.assertIn('X_train', heavy_data)
        self.assertIn('targets', heavy_data)
        # Shape check: (num_samples, seq_len, num_features)
        self.assertEqual(heavy_data['X_train'].shape[1], seq_len)
        self.assertEqual(heavy_data['X_train'].shape[2], len(self.preparer.feature_columns))
        
        # Check if target sequences are built correctly
        self.assertIn('target_bin', heavy_data['targets'])
        y_train_heavy = heavy_data['targets']['target_bin']['y_train']
        # The number of labels should match the number of sequences
        self.assertEqual(len(y_train_heavy), len(heavy_data['X_train']))

    def test_data_split(self):
        """Verify that the time-series split is done correctly."""
        data_packages = self.preparer.prepare_data_for_models(self.df)
        light_data = data_packages['light_models']
        
        test_size = mock_config['data.processing.preparation']['test_size']
        # The split is applied after dropping NaNs, so we calculate expected size
        df_clean = self.df.dropna(subset=self.preparer.feature_columns)
        df_clean = self.preparer.target_generator.generate_targets(df_clean)
        df_clean = df_clean.dropna(subset=self.preparer.feature_columns + ['target_reg', 'target_bin'])
        
        expected_train_len = int(len(df_clean) * (1 - test_size))
        self.assertEqual(len(light_data['X_train']), expected_train_len)


if __name__ == "__main__":
    unittest.main()
