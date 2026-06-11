"""
Analytics Math Utilities

Provides centralized, state-less mathematical functions for analytics components.
Reduces code duplication between Calculators and Analyzers.
"""

import numpy as np
import pandas as pd


def calculate_market_regime_metrics(prices: pd.Series, window: int = 20) -> dict[str, float]:
    """Calculate normalized market regime metrics."""
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window).std().iloc[-1]
    trend = returns.rolling(window).mean().iloc[-1]

    return {
        'volatility': float(volatility) if not pd.isna(volatility) else 0.0,
        'trend': float(trend) if not pd.isna(trend) else 0.0
    }

def calculate_herfindahl_hirschman_index(weights: np.ndarray) -> float:
    """Calculate HHI index."""
    if len(weights) < 1:
        return 0.0
    # Normalize weights
    weights = weights / np.sum(weights)
    return float(np.sum(weights ** 2))

def calculate_diversification_ratio(returns: pd.DataFrame, weights: np.ndarray) -> float:
    """Calculate diversification ratio."""
    # Simplified version for robust usage
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    weighted_vol = np.sum(weights * returns.std())
    return float(weighted_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
