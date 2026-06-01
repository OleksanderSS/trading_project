import numpy as np
import pandas as pd

from src.models.feature_selector import ModelFeatureSelector


def test_feature_leakage_prevention():
    selector = ModelFeatureSelector()
    cols = [f'f{i}' for i in range(45)] + ['target_leak', 'target_val']
    df = pd.DataFrame(np.random.rand(10, 47), columns=cols)
    
    # Run selector
    selected = selector.select_features(df)
    
    # Check that no target columns exist in selected features
    for col in selected.columns:
        assert not str(col).startswith('target_'), f"Leak detected in column {col}"
