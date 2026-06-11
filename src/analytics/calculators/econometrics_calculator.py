"""
Econometrics Calculator Proxy
Delegates core calculation to the unified FinancialMetricsLibrary.
"""

import pandas as pd

from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary


class EconometricsCalculator:
    """Proxy class for econometric calculations."""

    @staticmethod
    def run_granger_test(df: pd.DataFrame, target_col: str, predictor_cols: list[str], maxlag: int = 5) -> dict:
        """Proxies to run_granger_causality in the library for each predictor."""
        results = {}
        for col in predictor_cols:
            results[col] = FinancialMetricsLibrary.run_granger_causality(df, target_col, col, maxlag)
        return results

    @staticmethod
    def run_advanced_granger_test(df: pd.DataFrame, target_col: str, predictor_cols: list[str],
                                  maxlag: int = 10, lag_selection: str = 'aic') -> dict:
        """Advanced Granger test proxy."""
        # For now, reuse the unified granger test
        return EconometricsCalculator.run_granger_test(df, target_col, predictor_cols, maxlag)

    @staticmethod
    def get_var_forecast(df: pd.DataFrame, target_cols: list[str], steps: int = 5, **kwargs) -> pd.DataFrame:
        """Proxies to get_var_forecast in the library."""
        return FinancialMetricsLibrary.get_var_forecast(df, target_cols, steps)
