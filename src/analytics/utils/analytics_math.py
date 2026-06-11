"""
Analytics Math Utilities

Provides centralized, state-less mathematical functions for analytics components.
Reduces code duplication between Calculators and Analyzers.
"""

import numpy as np
import pandas as pd


def calculate_market_regime_metrics(prices: pd.Series, window: int = 20) -> dict[str, float]:
    """Calculate normalized market regime metrics."""
    if prices.empty or len(prices) < window:
        return {"volatility": 0.0, "trend": 0.0}

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"volatility": 0.0, "trend": 0.0}

    # Use shift(1) to avoid look-ahead bias
    volatility = returns.rolling(window, min_periods=1).std().shift(1).iloc[-1]
    trend = returns.rolling(window, min_periods=1).mean().shift(1).iloc[-1]

    return {
        "volatility": float(volatility) if not pd.isna(volatility) else 0.0,
        "trend": float(trend) if not pd.isna(trend) else 0.0,
    }


def calculate_herfindahl_hirschman_index(weights: np.ndarray) -> float:
    """Calculate HHI index."""
    if len(weights) < 1:
        return 0.0

    # Handle NaN values
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize weights
    total = np.sum(weights)
    if total == 0:
        return 0.0

    weights = weights / total
    return float(np.sum(weights**2))


def calculate_diversification_ratio(returns: pd.DataFrame, weights: np.ndarray) -> float:
    """Calculate diversification ratio."""
    if returns.empty or len(weights) < 1:
        return 1.0

    returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns_clean.empty or len(returns_clean) < 2:
        return 1.0

    weights_clean = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if len(weights_clean) != returns_clean.shape[1]:
        return 1.0

    # Simplified version for robust usage
    covariance = returns_clean.cov()
    portfolio_variance = np.dot(weights_clean.T, np.dot(covariance, weights_clean))
    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0:
        return 1.0

    portfolio_vol = np.sqrt(portfolio_variance)
    weighted_vol = np.sum(weights_clean * returns_clean.std())
    if not np.isfinite(weighted_vol) or not np.isfinite(portfolio_vol) or portfolio_vol <= 0:
        return 1.0

    return float(weighted_vol / portfolio_vol)
