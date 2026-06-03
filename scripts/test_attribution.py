import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.analyzers.performance_attribution_analyzer import PerformanceAttributionAnalyzer


def test_attribution():
    analyzer = PerformanceAttributionAnalyzer()
    
    dates = pd.date_range("2024-01-01", periods=100)
    portfolio_returns = pd.DataFrame(np.random.randn(100, 1) * 0.01, index=dates, columns=['Portfolio'])
    benchmark_returns = pd.DataFrame(np.random.randn(100, 1) * 0.01, index=dates, columns=['SPY'])
    
    data = {
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns
    }
    
    results = analyzer.analyze(data)
    print("Attribution Results Summary:")
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        for k, v in results.get('summary', {}).items():
            print(f"  {k}: {v}")
            
if __name__ == "__main__":
    test_attribution()
