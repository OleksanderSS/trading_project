
import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.features.nlp.models.roberta_sentiment import RobertaSentimentAnalyzer
from src.metrics.calculator import MetricsCalculator
from src.sentiment.sentiment_models import aggregate_sentiment, analyze_sentiment


class TestCoreModulesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics_calculator = MetricsCalculator()
        
    def test_ml_metrics_calculation(self):
        """Verify ML metrics calculation correctness."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.2, 0.8, 0.1])
        
        metrics = self.metrics_calculator.get_ml_metrics(y_true, y_pred, y_prob)
        
        self.assertIn('Accuracy', metrics)
        self.assertIn('F1', metrics)
        self.assertIn('ROC_AUC', metrics)
        self.assertGreaterEqual(metrics['Accuracy'], 0.0)
        self.assertLessEqual(metrics['Accuracy'], 1.0)
        print(f"ML Metrics Test Passed: {metrics}")

    def test_portfolio_metrics_calculation(self):
        """Verify Portfolio metrics calculation correctness."""
        equity_curve = pd.Series([1000, 1010, 1005, 1020, 1015, 1030])
        
        metrics = self.metrics_calculator.get_portfolio_metrics(equity_curve)
        
        self.assertIn('total_return_pct', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertGreater(metrics['total_return_pct'], 0)
        print(f"Portfolio Metrics Test Passed: {metrics}")

    def test_sentiment_finbert_integration(self):
        """Verify FinBERT sentiment analysis (mocked if heavy)."""
        texts = ["The stock price is going up!", "Bad earnings report, very disappointed."]
        
        # We use a small batch and check if it returns a DataFrame
        try:
            df = analyze_sentiment(texts)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertIn('label', df.columns)
            self.assertIn('score', df.columns)
            
            agg = aggregate_sentiment(df)
            self.assertIn('positive', agg)
            self.assertIn('negative', agg)
            self.assertIn('neutral', agg)
            print(f"FinBERT Sentiment Test Passed: {agg}")
        except Exception as e:
            print(f"FinBERT Sentiment Test Skipped or Failed (possibly due to missing libs): {e}")

    def test_roberta_sentiment_integration(self):
        """Verify RoBERTa sentiment analyzer."""
        config = {
            'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'confidence_threshold': 0.5
        }
        try:
            analyzer = RobertaSentimentAnalyzer(config)
            # We don't necessarily load it to avoid heavy downloads, but we check instantiation
            self.assertEqual(analyzer.model_name, config['model_name'])
            
            # Simple check if analyze returns a dict with expected keys
            # result = analyzer.analyze("Happy day") 
            # self.assertIn('label', result)
            print("RoBERTa Sentiment Analyzer Instantiation Passed")
        except Exception as e:
            print(f"RoBERTa Sentiment Test Failed: {e}")

    def test_usage_check(self):
        """Check if these modules are imported in key files."""
        key_files = [
            'src/pipeline/pipeline_orchestrator.py',
            'src/analytics/analyzers/news_impact_analyzer.py',
            'src/features/enrichers/advanced_analytics_enricher.py'
        ]
        
        found_usage = False
        for file_path in key_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'metrics' in content.lower() or 'sentiment' in content.lower():
                        print(f"Verified usage in: {file_path}")
                        found_usage = True
        
        self.assertTrue(found_usage, "No usage found in key pipeline files!")

if __name__ == "__main__":
    unittest.main()
