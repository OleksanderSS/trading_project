import numpy as np
import pandas as pd
from typing import Union

def safe_div(numerator: Union[float, np.ndarray, pd.Series], 
             denominator: Union[float, np.ndarray, pd.Series], 
             fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Performs safe division to prevent division by zero, returning fill_value.
    """
    if isinstance(denominator, (np.ndarray, pd.Series)):
        result = np.divide(numerator, denominator, out=np.full_like(denominator, fill_value, dtype=float), where=denominator != 0)
        return result
    return numerator / denominator if denominator != 0 else fill_value

def safe_sharpe(returns: pd.Series, risk_free: float = 0.0, annualization_factor: float = np.sqrt(252)) -> float:
    """
    Calculates Sharpe Ratio safely, handling zero standard deviation.
    """
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() - risk_free) / std * annualization_factor)

def safe_std(series: pd.Series, min_periods: int = 1) -> pd.Series:
    """
    Calculates rolling standard deviation safely.
    """
    return series.std() if len(series) >= min_periods else 0.0
