import numpy as np


def safe_sqrt(x: np.ndarray | float, epsilon: float = 1e-8) -> np.ndarray | float:
    """
    Safely computes square root by ensuring input is non-negative and replacing negative values with epsilon or 0.
    """
    return np.sqrt(np.maximum(x, epsilon))

def safe_log(x: np.ndarray | float, epsilon: float = 1e-8) -> np.ndarray | float:
    """
    Safely computes natural logarithm by ensuring input is positive.
    """
    return np.log(np.maximum(x, epsilon))
