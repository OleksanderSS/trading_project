import logging

import pandas as pd

from src.analytics.analyzers.performance_attribution_analyzer import PerformanceAttributionAnalyzer

# Налаштування логування
logging.basicConfig(level=logging.INFO)

def run_debug():
    analyzer = PerformanceAttributionAnalyzer()
    
    # Створення тестових даних
    index = pd.date_range("2023-01-01", periods=5)
    portfolio = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01, 0.03, 0.01],
        "TSLA": [0.05, -0.03, 0.02, 0.01, -0.02]
    }, index=index)
    
    benchmark = pd.DataFrame({
        "SPY": [0.01, 0.01, 0.00, 0.02, 0.01]
    }, index=index)
    
    data = {
        "portfolio_returns": portfolio,
        "benchmark_returns": benchmark,
        "weights": {"AAPL": 0.6, "TSLA": 0.4}
    }
    
    result = analyzer.analyze(data)
    print("\nResult keys:", result.keys())

if __name__ == "__main__":
    run_debug()
