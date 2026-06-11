import pandas as pd
import pytest
import numpy as np

def test_macro_merging_leakage():
    """
    Test that merging macro data and using bfill() causes lookahead leakage,
    while ffill() preserves causality.
    """
    # Simulate daily price data
    price_df = pd.DataFrame({
        'price': [100, 101, 102, 103]
    }, index=pd.date_range('2026-01-01', periods=4))

    # Simulate monthly macro data (only available on 2026-01-03)
    macro_df = pd.DataFrame({
        'macro': [5.0]
    }, index=pd.to_datetime(['2026-01-03']))

    # Merge
    df = price_df.join(macro_df, how='left')

    # Case 1: bfill() (Current problematic approach)
    df_bfill = df.copy().bfill()
    # Leakage check: data for 2026-01-01 now has macro value from 2026-01-03
    assert df_bfill.loc['2026-01-01', 'macro'] == 5.0
    
    # Case 2: ffill() (Correct causal approach)
    df_ffill = df.copy().ffill()
    # Causal check: 2026-01-01 should be NaN if macro not yet available
    assert pd.isna(df_ffill.loc['2026-01-01', 'macro'])
    # 2026-01-03 should have the value
    assert df_ffill.loc['2026-01-03', 'macro'] == 5.0

if __name__ == "__main__":
    pytest.main([__file__])
