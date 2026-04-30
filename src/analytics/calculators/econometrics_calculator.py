"""
Provides a collection of econometric and statistical tools for financial analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

logger = logging.getLogger(__name__)

class EconometricsCalculator:
    """A collection of static methods for performing econometric tests and models."""

    @staticmethod
    def run_granger_test(df: pd.DataFrame, target_col: str, predictor_cols: List[str], maxlag: int = 5) -> Dict[str, Any]:
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
    def _test_single_predictor(df: pd.DataFrame, target_col: str, predictor_col: str, maxlag: int) -> Dict[str, Any]:
        """Test Granger causality for a single predictor variable."""
        if not EconometricsCalculator._validate_columns(df, target_col, predictor_col):
            return None
            
        test_data = df[[target_col, predictor_col]].dropna()
        if not EconometricsCalculator._validate_data_length(test_data, maxlag, predictor_col):
            return None

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
        return min(p_values)

    @staticmethod
    def _build_causality_result(predictor_col: str, target_col: str, correlation: float, p_value: float) -> Dict[str, Any]:
        """Build causality result and check for spurious correlation."""
        significant = p_value < 0.05
        result = {"p_value": p_value, "is_causal": significant, "correlation": correlation}
        
        if abs(correlation) > 0.7 and not significant:
            logger.warning(f"Spurious Correlation Alert: {predictor_col} and {target_col} are highly correlated ({correlation:.2f}) but lack Granger causality (p={p_value:.4f}).")
        
        return result

    @staticmethod
    def get_var_forecast(df: pd.DataFrame, target_cols: List[str], steps: int = 5, maxlags: int = 15, ic='aic') -> pd.DataFrame:
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
    def _generate_var_forecast(df: pd.DataFrame, target_cols: List[str], config: Dict[str, Any]) -> pd.DataFrame:
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
    def _fit_var_and_forecast(model_data: pd.DataFrame, valid_cols: List[str], config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Fit VAR model and generate forecast."""
        model = VAR(model_data)
        results = model.fit(maxlags=config['maxlags'], ic=config['ic'])
        
        last_obs = model_data.values[-results.k_ar:]
        return results.forecast(y=last_obs, steps=config['steps'])

    @staticmethod
    def _create_forecast_dataframe(forecast: np.ndarray, valid_cols: List[str], last_index, steps: int) -> pd.DataFrame:
        """Create forecast DataFrame with proper index."""
        return pd.DataFrame(
            forecast, 
            columns=valid_cols, 
            index=pd.date_range(start=last_index + pd.Timedelta(days=1), periods=steps)
        )
