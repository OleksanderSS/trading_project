import unittest
import pandas as pd
import numpy as np
from src.features.enrichers.technical_analysis_enricher import TechnicalAnalysisEnricher

class TestTechnicalAnalysisEnricher(unittest.TestCase):
    def setUp(self):
        self.enricher = TechnicalAnalysisEnricher()
        # Create dummy data
        dates = pd.date_range('2023-01-01', periods=100)
        self.data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_enrich_hurst_exponent(self):
        # Test Hurst exponent calculation stability
        # Hurst exponent is added in _add_advanced_features via _add_econometrics_features
        # We need to ensure the enricher can handle very short data and data with 0 variance
        
        # 1. Normal data
        enriched = self.enricher.enrich(self.data)
        self.assertIn('HURST_EXPONENT', enriched.columns)
        self.assertTrue(np.isfinite(enriched['HURST_EXPONENT']).all())
        
        # 2. Short data (should return default 0.5)
        short_data = self.data.iloc[:3]
        enriched_short = self.enricher.enrich(short_data)
        self.assertIn('HURST_EXPONENT', enriched_short.columns)
        self.assertTrue((enriched_short['HURST_EXPONENT'] == 0.5).all())
        
        # 3. Constant data (0 variance)
        constant_data = self.data.copy()
        constant_data['close'] = 100.0
        enriched_const = self.enricher.enrich(constant_data)
        self.assertIn('HURST_EXPONENT', enriched_const.columns)
        self.assertTrue(np.isfinite(enriched_const['HURST_EXPONENT']).all())

    def test_advanced_features_stability(self):
        # Test stability of other advanced features
        enriched = self.enricher.enrich(self.data)
        expected_cols = [
            'VOLATILITY_5', 'VOLATILITY_20', 'VOLATILITY_50',
            'MOMENTUM_ZSCORE', 'SHARPE_RATIO', 'SORTINO_RATIO',
            'AUTOCORR', 'SKEWNESS', 'KURTOSIS'
        ]
        for col in expected_cols:
            self.assertIn(col, enriched.columns)
            # Some metrics might be NaN at the beginning of the series, but should be finite later
            self.assertTrue(np.isfinite(enriched[col].iloc[-10:]).all())

if __name__ == '__main__':
    unittest.main()
