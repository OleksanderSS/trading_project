import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def test_train_test_split_temporal_integrity():
    """
    Verify that train_test_split with shuffle=False
    preserves the temporal order of the data.
    """
    # Create dummy time-series data
    n_samples = 100
    df = pd.DataFrame({
        'feature': np.random.randn(n_samples),
        'target': np.arange(n_samples) # Represents time/sequence
    })
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Assertions
    # Train set should contain the first 80 samples
    assert len(X_train) == 80
    assert len(X_test) == 20
    
    # Check that train contains smaller indices than test
    assert y_train.max() < y_test.min()
    
    # Check that sequences are preserved
    assert y_train.iloc[0] == 0
    assert y_train.iloc[-1] == 79
    assert y_test.iloc[0] == 80
    assert y_test.iloc[-1] == 99

if __name__ == "__main__":
    pytest.main([__file__])
