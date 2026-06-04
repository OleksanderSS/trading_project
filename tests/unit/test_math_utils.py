import unittest
import numpy as np
from src.core.utils.math_utils import safe_sqrt, safe_log

class TestMathUtils(unittest.TestCase):
    def test_safe_sqrt(self):
        # Positive values
        self.assertAlmostEqual(safe_sqrt(4.0), 2.0)
        self.assertAlmostEqual(safe_sqrt(0.0, epsilon=1e-8), np.sqrt(1e-8))
        
        # Negative values
        self.assertAlmostEqual(safe_sqrt(-1.0, epsilon=1e-8), np.sqrt(1e-8))
        
        # Array inputs
        arr = np.array([4.0, 0.0, -1.0])
        result = safe_sqrt(arr, epsilon=1e-8)
        np.testing.assert_array_almost_equal(result, [2.0, np.sqrt(1e-8), np.sqrt(1e-8)])

    def test_safe_log(self):
        # Positive values
        self.assertAlmostEqual(safe_log(1.0), 0.0)
        self.assertAlmostEqual(safe_log(np.e), 1.0)
        
        # Zero and negative values
        self.assertAlmostEqual(safe_log(0.0, epsilon=1e-8), np.log(1e-8))
        self.assertAlmostEqual(safe_log(-1.0, epsilon=1e-8), np.log(1e-8))
        
        # Array inputs
        arr = np.array([1.0, 0.0, -1.0])
        result = safe_log(arr, epsilon=1e-8)
        np.testing.assert_array_almost_equal(result, [0.0, np.log(1e-8), np.log(1e-8)])

if __name__ == '__main__':
    unittest.main()
