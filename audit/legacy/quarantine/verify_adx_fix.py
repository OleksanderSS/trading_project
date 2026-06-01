import numpy as np
from src.algorithms.regime.metrics import RegimeMetricsCalculator

def test_adx_error_handling():
    # Test with empty returns to trigger the exception
    returns = np.array([0.1])
    try:
        adx = RegimeMetricsCalculator._calculate_adx(returns, period=14)
        print(f"ADX result: {adx}")
        # Expected result is 0.0, and it should log an error (check logs manually if needed)
    except Exception as e:
        print(f"Test failed with exception: {e}")

if __name__ == "__main__":
    test_adx_error_handling()
