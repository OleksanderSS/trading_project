import pandas as pd

from src.models.adapters.categorical_helper import handle_categorical_features_split


def test_leakage_prevention():
    # Setup dummy data
    train = pd.DataFrame({'cat': ['A', 'B', 'A'], 'val': [1, 2, 3]})
    val = pd.DataFrame({'cat': ['A', 'C', 'B'], 'val': [4, 5, 6]})
    test = pd.DataFrame({'cat': ['C', 'A', 'B'], 'val': [7, 8, 9]})
    
    tr, v, t, info = handle_categorical_features_split(train, val, test, exclude_cols=[])
    
    print("Train encoded:\n", tr)
    print("Val encoded:\n", v)
    print("Test encoded:\n", t)
    
    # 'C' was unseen in training, should be 0 (False for cat_B)
    assert v.loc[1, 'cat_B'] == 0
    assert t.loc[0, 'cat_B'] == 0
    print("✅ Leakage test passed!")

if __name__ == "__main__":
    test_leakage_prevention()
