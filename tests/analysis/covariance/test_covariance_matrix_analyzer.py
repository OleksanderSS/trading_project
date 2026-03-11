
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Assuming the file is in src.analytics.covariance
from src.analytics.covariance.covariance_matrix_analyzer import CovarianceMatrixAnalyzer

class TestCovarianceMatrixAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and test data for the tests."""
        # Create a mock for the data fetcher
        self.mock_data_fetcher = MagicMock()
        # Create a mock for the data preprocessor
        self.mock_data_preprocessor = MagicMock()

        # Sample data to be returned by the fetcher
        self.sample_raw_data = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'ticker': ['AAPL', 'AAPL', 'AAPL'],
            'close': [150.0, 152.0, 151.0]
        })

        # Sample data to be returned by the preprocessor
        self.sample_processed_data = pd.DataFrame({
            'AAPL': [0.0133, -0.0066],
            'MSFT': [0.01, -0.005]
        })

        # Configure the mocks to return the sample data
        self.mock_data_fetcher.fetch_data.return_value = self.sample_raw_data
        self.mock_data_preprocessor.preprocess_for_covariance.return_value = self.sample_processed_data

        # Tickers and date range for the analyzer
        self.tickers = ['AAPL', 'MSFT']
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-03'

    def test_analyze_returns_covariance_and_correlation_matrix(self):
        """Test that the analyze method returns the expected dictionary structure."""
        # Initialize the analyzer with the mocks
        analyzer = CovarianceMatrixAnalyzer(
            data_fetcher=self.mock_data_fetcher,
            data_preprocessor=self.mock_data_preprocessor,
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Run the analysis
        result = analyzer.analyze()

        # --- Assertions ---
        # Check that the fetch_data method was called correctly
        self.mock_data_fetcher.fetch_data.assert_called_once_with(
            self.tickers, self.start_date, self.end_date
        )

        # Check that the preprocess_for_covariance method was called correctly
        self.mock_data_preprocessor.preprocess_for_covariance.assert_called_once_with(
            self.sample_raw_data
        )

        # Check that the result contains the expected keys
        self.assertIn('covariance_matrix', result)
        self.assertIn('correlation_matrix', result)

        # Check that the matrices are pandas DataFrames
        self.assertIsInstance(result['covariance_matrix'], pd.DataFrame)
        self.assertIsInstance(result['correlation_matrix'], pd.DataFrame)

        # Check the dimensions of the matrices
        self.assertEqual(result['covariance_matrix'].shape, (2, 2))
        self.assertEqual(result['correlation_matrix'].shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
