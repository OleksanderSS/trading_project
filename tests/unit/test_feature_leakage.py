import numpy as np
import pandas as pd

from src.archive.models.feature_selector import ModelFeatureSelector


def test_feature_leakage_prevention():
    selector = ModelFeatureSelector()
    cols = [f'f{i}' for i in range(45)] + [
        'target_leak',
        'target_val',
        'TARGET_RETURN_1P',
        'state_TARGET_RETURN_1P',
    ]
    df = pd.DataFrame(np.random.rand(10, len(cols)), columns=cols)
    
    # Run selector
    selected = selector.select_features(df)
    
    # Check that no target columns exist in selected features
    for col in selected.columns:
        assert 'target' not in str(col).lower(), f"Leak detected in column {col}"
