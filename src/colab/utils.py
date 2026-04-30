"""Utility functions for Colab training."""

import hashlib
from functools import wraps
import time
from typing import Dict, Any


def retry_on_timeout(max_retries=3, wait_seconds=5):
    """Decorator for retry on timeout."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    if attempt < max_retries - 1:
                        print(
                            f"⚠️ Attempt {attempt + 1} failed: "
                            f"{str(e)[:100]}")
                        print(f"   Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                    else:
                        print(f"❌ Failed after {max_retries} attempts")
                        raise
        return wrapper
    return decorator


def compute_data_signature(df_feat, df_targ) -> str:
    """Compute data signature for caching."""
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


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics."""
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    else:
        mape = 0.0
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }
