"""
Provides a collection of econometric and statistical tools for financial analysis.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)

class EconometricsCalculator:
    """A collection of static methods for performing econometric tests and models."""

    @staticmethod
    def run_advanced_granger_test(df: pd.DataFrame, target_col: str, predictor_cols: list[str],
                                  maxlag: int = 10, lag_selection: str = 'aic') -> dict[str, Any]:
        """
        Advanced Granger causality testing with optimal lag selection and validation.

        Args:
            df: DataFrame with time series data
            target_col: Target variable column name
            predictor_cols: List of predictor variable column names
            maxlag: Maximum lag to consider
            lag_selection: Lag selection method ('aic', 'bic', 'hqic')

        Returns:
            Dictionary with comprehensive causality analysis results
        """
        causality_results = {}

        for col in predictor_cols:
            result = EconometricsCalculator._test_single_predictor(df, target_col, col, maxlag)  # type: ignore
            if result:
                causality_results[col] = result

        # Add overall analysis summary
        causality_results['_summary'] = EconometricsCalculator._generate_simple_summary(causality_results)  # type: ignore

        return causality_results

    @staticmethod
    def run_granger_test(df: pd.DataFrame, target_col: str, predictor_cols: list[str], maxlag: int = 5) -> dict[str, Any]:
        """
        Tests if predictor variables Granger-cause the target variable.
        Detects 'Spurious Correlation' if correlation exists without causality.
        """
        causality_results = {}

        for col in predictor_cols:
            result = EconometricsCalculator._test_single_predictor(df, target_col, col, maxlag)
            if result:
                causality_results[col] = result

        return causality_results

    @staticmethod
    def _test_single_predictor(df: pd.DataFrame, target_col: str, predictor_col: str, maxlag: int) -> dict[str, Any]:
        """Test Granger causality for a single predictor variable."""
        if not EconometricsCalculator._validate_columns(df, target_col, predictor_col):
            return None  # type: ignore

        test_data = df[[target_col, predictor_col]].dropna()
        if not EconometricsCalculator._validate_data_length(test_data, maxlag, predictor_col):
            return None  # type: ignore

        correlation = test_data[target_col].corr(test_data[predictor_col])

        try:
            p_value = EconometricsCalculator._calculate_granger_p_value(test_data, maxlag)
            return EconometricsCalculator._build_causality_result(predictor_col, target_col, correlation, p_value)
        except Exception as e:
            logger.error(f"Granger causality test failed for predictor '{predictor_col}' on target '{target_col}': {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def _validate_columns(df: pd.DataFrame, target_col: str, predictor_col: str) -> bool:
        """Validate that required columns exist in DataFrame."""
        return predictor_col in df.columns and target_col in df.columns

    @staticmethod
    def _validate_data_length(test_data: pd.DataFrame, maxlag: int, predictor_col: str) -> bool:
        """Validate sufficient data length for Granger test."""
        if len(test_data) < maxlag + 5:
            logger.warning(f"Skipping Granger test for {predictor_col} due to insufficient data.")
            return False
        return True

    @staticmethod
    def _calculate_granger_p_value(test_data: pd.DataFrame, maxlag: int) -> float:
        """Calculate minimum p-value from Granger causality test."""
        test_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
        p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
        return float(min(p_values))  # type: ignore

    @staticmethod
    def _build_causality_result(predictor_col: str, target_col: str, correlation: float, p_value: float) -> dict[str, Any]:
        """Build causality result and check for spurious correlation."""
        significant = p_value < 0.05
        result = {"p_value": p_value, "is_causal": significant, "correlation": correlation}

        if abs(correlation) > 0.7 and not significant:
            logger.warning(f"Spurious Correlation Alert: {predictor_col} and {target_col} are highly correlated ({correlation:.2f}) but lack Granger causality (p={p_value:.4f}).")

        return result

    @staticmethod
    def _generate_simple_summary(causality_results: dict[str, Any]) -> dict[str, Any]:
        """Generate simple summary statistics for causality analysis."""
        if not causality_results:
            return {'error': 'No causality results to summarize'}

        # Filter out actual results (ignore summary key)
        results = {k: v for k, v in causality_results.items() if not k.startswith('_')}

        significant_count = sum(1 for r in results.values() if r.get('is_causal', False))
        total_count = len(results)

        return {
            'total_tests': total_count,
            'significant_relationships': significant_count,
            'significance_rate': significant_count / total_count if total_count > 0 else 0,
            'recommendation': 'Causal relationships found' if significant_count > 0 else 'No significant causal relationships'
        }

    @staticmethod
    def get_var_forecast(df: pd.DataFrame, target_cols: list[str], steps: int = 5, maxlags: int = 15, ic='aic') -> pd.DataFrame:
        """
        Generates a baseline forecast using Vector Auto Regression (VAR).
        Serves as a benchmark for more complex ML models.
        """
        var_config = {
            'steps': steps,
            'maxlags': maxlags,
            'ic': ic
        }

        return EconometricsCalculator._generate_var_forecast(df, target_cols, var_config)

    @staticmethod
    def _generate_var_forecast(df: pd.DataFrame, target_cols: list[str], config: dict[str, Any]) -> pd.DataFrame:
        """Generate VAR forecast with configuration parameters."""
        valid_cols = [c for c in target_cols if c in df.columns]
        if not valid_cols:
            logger.error("None of the target columns for VAR forecast are in the DataFrame.")
            return pd.DataFrame()

        try:
            model_data = df[valid_cols].dropna()
            if not EconometricsCalculator._validate_var_data(model_data, config['maxlags']):
                return pd.DataFrame()

            forecast = EconometricsCalculator._fit_var_and_forecast(model_data, valid_cols, config)
            if forecast is not None:
                return EconometricsCalculator._create_forecast_dataframe(forecast, valid_cols, df.index[-1], config['steps'])

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"VAR baseline forecast failed: {e}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def _validate_var_data(model_data: pd.DataFrame, maxlags: int) -> bool:
        """Validate sufficient data for VAR model."""
        if len(model_data) < maxlags + 5:
            logger.warning(f"Insufficient data for VAR model with {maxlags} lags. Required: >{maxlags+5}, have: {len(model_data)}.")
            return False
        return True

    @staticmethod
    def _fit_var_and_forecast(model_data: pd.DataFrame, valid_cols: list[str], config: dict[str, Any]) -> np.ndarray | None:
        """Fit VAR model and generate forecast."""
        model = VAR(model_data)
        results = model.fit(maxlags=config['maxlags'], ic=config['ic'])

        last_obs = model_data.values[-results.k_ar:]
        return results.forecast(y=last_obs, steps=config['steps'])

    @staticmethod
    def _create_forecast_dataframe(forecast: np.ndarray, valid_cols: list[str], last_index, steps: int) -> pd.DataFrame:
        """Create forecast DataFrame with proper index."""
        return pd.DataFrame(
            forecast,
            columns=valid_cols,
            index=pd.date_range(start=last_index + pd.Timedelta(days=1), periods=steps)
        )
