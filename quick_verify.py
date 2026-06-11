
import os
import sys

import numpy as np
import pandas as pd

# Add project root
sys.path.append(os.getcwd())

from src.metrics.calculator import MetricsCalculator
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary


def test_metrics_logic():
    print("--- Starting Metrics Logic Tests ---")
    
    # 1. Test FinancialMetricsLibrary directly (the math engine)
    equity_curve = pd.Series([100, 110, 105, 120, 115])
    print(f"Equity Curve: {equity_curve.tolist()}")
    
    cagr = FinancialMetricsLibrary.calculate_cagr(equity_curve)
    print(f"CAGR: {cagr:.4f}")
    
    dd = FinancialMetricsLibrary.calculate_max_drawdown(equity_curve)
    print(f"Max Drawdown: {dd:.4f}")
    
    # 2. Test MetricsCalculator (the integration interface)
    calc = MetricsCalculator()
    
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 0])
    
    ml_report = calc.get_ml_metrics(y_true, y_pred)
    print(f"ML Metrics: {ml_report}")
    
    portfolio_report = calc.get_portfolio_metrics(equity_curve)
    print(f"Portfolio Metrics Keys: {list(portfolio_report.keys())}")
    
    print("--- Metrics Logic Tests Complete ---")

if __name__ == "__main__":
    try:
        test_metrics_logic()
    except Exception as e:
        print(f"Test failed: {e}")
