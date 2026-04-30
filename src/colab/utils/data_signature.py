"""Data signature computation for caching"""

import hashlib


def compute_data_signature(df_feat, df_targ):
    """Обчислити сигнатуру даних для кешування"""
    import pandas as pd
    feat_info = (
        f"{df_feat.shape}_"
        f"{pd.util.hash_pandas_object(df_feat.tail(100)).sum()}"
    )
    targ_info = (
        f"{df_targ.shape}_"
        f"{pd.util.hash_pandas_object(df_targ.tail(100)).sum()}"
    )
    combined = f"{feat_info}_{targ_info}"
    return hashlib.sha256(combined.encode()).hexdigest()
