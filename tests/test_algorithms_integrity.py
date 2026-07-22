import numpy as np
import pandas as pd

from src.analytics.detectors.bias_detector import BiasDetector
from src.algorithms.risk_parity_allocator import RiskParityAllocator
from src.algorithms.walk_forward_optimizer import WalkForwardOptimizer


def test_risk_parity_integrity():
    print("Testing Risk Parity Allocator integrity...")
    
    allocator = RiskParityAllocator()
    assets = ['AAPL', 'GOOG', 'MSFT']
    vols = {'AAPL': 0.2, 'GOOG': 0.25, 'MSFT': 0.15}
    # Identity correlation for simplicity
    correlations = np.array([
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Test 1: Allocation
    result = allocator.allocate(assets, vols, correlations)
    assert 'weights' in result, "Weights not found in result"
    print(f"Allocation successful: {result['weights']}")
    
    # Test 2: Portfolio Optimization (exposed method)
    opt_result = allocator.optimize_portfolio(assets, vols, correlations, {'objective': 'min_vol'})
    assert 'weights' in opt_result, "Optimization failed"
    print(f"Min Volatility Optimization successful: {opt_result['weights']}")


def test_bias_detector_flags_future_return_leakage():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04])
    prices = (100 * (1 + returns).cumprod()).to_frame("close")
    signals = prices.pct_change().shift(-1).rename(columns={"close": "close"})

    result = BiasDetector().detect_look_ahead_bias(signals, prices, threshold=0.99)

    assert result["lookahead_bias_detected"] is True
    assert result["suspicious_signals"]


def test_walk_forward_optimizer_runs_real_folds():
    prices = pd.DataFrame({"close": np.linspace(100.0, 130.0, 120)})
    optimizer = WalkForwardOptimizer()

    result = optimizer.run_walk_forward(
        prices,
        {"lookback": [10, 20], "threshold": [0.1, 0.2]},
        n_splits=3,
    )

    assert result["success"] is True
    assert result["fold_results"]
    assert set(result["best_params"]).issubset({"lookback", "threshold"})

if __name__ == "__main__":
    test_risk_parity_integrity()
