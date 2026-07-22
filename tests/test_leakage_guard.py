import pandas as pd
import numpy as np
import pytest
from src.features.selection.smart_selector import SmartFeatureSelector
from src.models.analysis.baselines.models import LinearRegressionBaseline, SimpleRandomForestBaseline

def test_leakage_protection_smart_selector():
    # Setup
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target_fake': [0, 1, 0],  # Should be dropped
        'feature2': [4, 5, 6]
    })
    target = pd.Series([0, 1, 0])
    selector = SmartFeatureSelector()
    
    # We expect this to run without error, and 'target_fake' should not be used
    # Since we can't easily introspect the model inside, we check if it runs.
    # If it fails, the leakage protection might have broken something.
    selected = selector.select(df, target, "test_context", is_classification=True)
    assert 'target_fake' not in selected

def test_leakage_protection_baselines():
    # Setup
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target_fake': [0, 1, 0],  # Should be dropped
    })
    target = pd.Series([1.0, 2.0, 3.0])
    
    # Test LinearRegressionBaseline
    lr = LinearRegressionBaseline()
    lr.train_and_evaluate(pd.DataFrame(), df, target)
    
    # Test SimpleRandomForestBaseline
    rf = SimpleRandomForestBaseline()
    rf.train_and_evaluate(pd.DataFrame(), df, target)

if __name__ == "__main__":
    pytest.main([__file__])
