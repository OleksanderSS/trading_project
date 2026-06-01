import os
import sys

import numpy as np
import pandas as pd

# Додаємо кореневу директорію проекту до sys.path
sys.path.append(os.getcwd())

from src.core.logging.logger import ProjectLogger
from src.metrics.calculator import MetricsCalculator
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary
from src.metrics.utils.calculation_tools import calculate_drawdown_series


def test_ml_metrics():
    logger = ProjectLogger.get_logger("TestMetrics")
    logger.info("--- Testing ML Metrics ---")
    calculator = MetricsCalculator()
    
    # Test Classification
    y_true_clf = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred_clf = [1, 0, 1, 0, 0, 1, 1, 0]
    y_prob_clf = [0.9, 0.1, 0.8, 0.4, 0.2, 0.85, 0.7, 0.3]
    
    clf_metrics = calculator.get_ml_metrics(y_true_clf, y_pred_clf, y_prob=y_prob_clf)
    logger.info(f"Classification Metrics: {clf_metrics}")
    assert "Accuracy" in clf_metrics
    assert "ROC_AUC" in clf_metrics
    
    # Test Regression
    y_true_reg = np.array([10.5, 11.2, 10.8, 12.1])
    y_pred_reg = np.array([10.4, 11.5, 10.7, 12.0])
    
    reg_metrics = calculator.get_ml_metrics(y_true_reg, y_pred_reg)
    logger.info(f"Regression Metrics: {reg_metrics}")
    assert "MAE" in reg_metrics
    assert "RMSE" in reg_metrics
    
    logger.info("ML Metrics tests passed!")

def test_financial_metrics():
    logger = ProjectLogger.get_logger("TestMetrics")
    logger.info("--- Testing Financial Metrics ---")
    calculator = MetricsCalculator()
    
    # Generate dummy equity curve (100 days of random returns)
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.01, 100)
    equity_curve = pd.Series(100000 * np.cumprod(1 + daily_returns))
    
    portfolio_metrics = calculator.get_portfolio_metrics(equity_curve)
    logger.info(f"Portfolio Metrics: {portfolio_metrics}")
    
    assert "sharpe_ratio" in portfolio_metrics
    assert "max_drawdown" in portfolio_metrics
    assert "cagr" in portfolio_metrics
    
    # Comparison between library and tools
    lib_dd = FinancialMetricsLibrary.calculate_drawdowns(equity_curve)
    tool_dd = calculate_drawdown_series(equity_curve)
    
    pd.testing.assert_series_equal(lib_dd, tool_dd, check_names=False)
    logger.info("Drawdown calculation consistency check passed!")
    
    logger.info("Financial Metrics tests passed!")

def test_full_report():
    logger = ProjectLogger.get_logger("TestMetrics")
    logger.info("--- Testing Full Report ---")
    calculator = MetricsCalculator()
    
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    equity_curve = pd.Series([100, 105, 102, 110])
    
    report = calculator.get_full_report(y_true=y_true, y_pred=y_pred, equity_curve=equity_curve)
    logger.info(f"Full Report Summary: {report['summary']}")
    
    assert "ml" in report
    assert "portfolio" in report
    assert "summary" in report
    assert report["summary"]["status"] == "success"
    
    logger.info("Full Report test passed!")

def test_edge_cases():
    logger = ProjectLogger.get_logger("TestMetrics")
    logger.info("--- Testing Edge Cases ---")
    calculator = MetricsCalculator()
    
    # Empty data
    empty_res = calculator.calculate(y_true=[], y_pred=[])
    logger.info(f"Empty Data Result: {empty_res}")
    assert empty_res["ml"] == {}
    
    # Data with NaNs
    y_true_nan = [1, np.nan, 1, 0]
    y_pred_nan = [1, 1, np.nan, 0]
    nan_res = calculator.get_ml_metrics(y_true_nan, y_pred_nan)
    logger.info(f"NaN Data Result: {nan_res}")
    # MLEvaluator clears NaNs, so it should still calculate something if valid pairs exist
    assert "Accuracy" in nan_res
    
    logger.info("Edge cases tests passed!")

if __name__ == "__main__":
    try:
        test_ml_metrics()
        test_financial_metrics()
        test_full_report()
        test_edge_cases()
        print("\nALL METRICS TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
