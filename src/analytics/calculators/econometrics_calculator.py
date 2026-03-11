"""
Provides a collection of econometric and statistical tools for financial analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
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
            if col not in df.columns or target_col not in df.columns:
                continue
            
            # Ensure there is enough data
            test_data = df[[target_col, col]].dropna()
            if len(test_data) < maxlag + 5:
                logger.warning(f"Skipping Granger test for {col} due to insufficient data.")
                continue

            correlation = test_data[target_col].corr(test_data[col])
            
            try:
                test_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
                min_p = min(p_values)
                
                significant = min_p < 0.05
                causality_results[col] = {"p_value": min_p, "is_causal": significant, "correlation": correlation}
                
                if abs(correlation) > 0.7 and not significant:
                    logger.warning(f"Spurious Correlation Alert: {col} and {target_col} are highly correlated ({correlation:.2f}) but lack Granger causality (p={min_p:.4f}).")
            
            except Exception as e:
                logger.error(f"Granger causality test failed for predictor '{col}' on target '{target_col}': {e}", exc_info=True)
                causality_results[col] = {"error": str(e)}
                
        return causality_results

    @staticmethod
    def get_var_forecast(df: pd.DataFrame, target_cols: List[str], steps: int = 5, maxlags: int = 15, ic='aic') -> pd.DataFrame:
        """
        Generates a baseline forecast using Vector Auto Regression (VAR).
        Serves as a benchmark for more complex ML models.
        """
        valid_cols = [c for c in target_cols if c in df.columns]
        if not valid_cols:
            logger.error("None of the target columns for VAR forecast are in the DataFrame.")
            return pd.DataFrame()

        try:
            model_data = df[valid_cols].dropna()
            if len(model_data) < maxlags + 5:
                logger.warning(f"Insufficient data for VAR model with {maxlags} lags. Required: >{maxlags+5}, have: {len(model_data)}.")
                return pd.DataFrame()

            model = VAR(model_data)
            results = model.fit(maxlags=maxlags, ic=ic)
            
            # The forecast needs the last `k_ar` observations
            last_obs = model_data.values[-results.k_ar:]
            forecast = results.forecast(y=last_obs, steps=steps)
            
            forecast_df = pd.DataFrame(forecast, columns=valid_cols, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps))
            return forecast_df
        
        except Exception as e:
            logger.error(f"VAR baseline forecast failed: {e}", exc_info=True)
            return pd.DataFrame()
