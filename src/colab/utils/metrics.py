"""Metrics computation for model evaluation"""


def compute_metrics(y_true, y_pred):
    """Розрахувати метрики якості моделі"""
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
        'mape': float(mape)}
