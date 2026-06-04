"""
Advanced econometric tools for comprehensive causal analysis.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


class AdvancedEconometricsCalculator:
    """Advanced econometric methods for robust causal analysis."""

    @staticmethod
    def run_comprehensive_causal_analysis(df: pd.DataFrame, target_col: str,
        predictor_cols: list[str], maxlag: int=10, lag_selection: str='aic'
        ) ->dict[str, Any]:
        """
        Comprehensive causal analysis with Granger tests, stationarity, cointegration, and validation.

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
            result = (AdvancedEconometricsCalculator.
                _test_single_predictor_comprehensive(df, target_col, col,
                maxlag, lag_selection))
            if result:
                causality_results[col] = result
        causality_results['_summary'
            ] = AdvancedEconometricsCalculator._generate_causality_summary(
            causality_results)
        return causality_results

    @staticmethod
    def _test_single_predictor_comprehensive(df: pd.DataFrame, target_col:
        str, predictor_col: str, maxlag: int, lag_selection: str) ->dict[
        str, Any]:
        """Comprehensive Granger causality test with optimal lag selection and validation."""
        if not AdvancedEconometricsCalculator._validate_columns(df,
            target_col, predictor_col):
            return {'error': 'Column validation failed'}
        test_data = df[[target_col, predictor_col]].dropna()
        if not AdvancedEconometricsCalculator._validate_data_length(test_data,
            maxlag, predictor_col):
            return {'error': 'Data length validation failed'}
        try:
            stationarity_results = (AdvancedEconometricsCalculator.
                _test_stationarity(test_data, target_col, predictor_col))
            optimal_lag = AdvancedEconometricsCalculator._select_optimal_lag(
                test_data, maxlag, lag_selection)
            granger_results = (AdvancedEconometricsCalculator.
                _run_granger_with_validation(test_data, optimal_lag))
            cointegration_results = None
            if stationarity_results[target_col]['is_i1'
                ] and stationarity_results[predictor_col]['is_i1']:
                cointegration_results = (AdvancedEconometricsCalculator.
                    _test_cointegration(test_data))
            impulse_response = (AdvancedEconometricsCalculator.
                _calculate_impulse_response(test_data, optimal_lag))
            variance_decomp = (AdvancedEconometricsCalculator.
                _calculate_variance_decomposition(test_data, optimal_lag))
            return {'predictor': predictor_col, 'target': target_col,
                'optimal_lag': optimal_lag, 'stationarity':
                stationarity_results, 'granger_test': granger_results,
                'cointegration': cointegration_results, 'impulse_response':
                impulse_response, 'variance_decomposition': variance_decomp,
                'is_significant': granger_results['p_value'] < 0.05,
                'causality_strength': AdvancedEconometricsCalculator.
                _calculate_causality_strength(granger_results,
                stationarity_results, cointegration_results or {})}
        except Exception as e:
            logger.error(
                f"Comprehensive Granger test failed for predictor '{predictor_col}' on target '{target_col}': {e}"
                , exc_info=True)
            return {'error': str(e)}

    @staticmethod
    def _test_stationarity(test_data: pd.DataFrame, target_col: str,
        predictor_col: str) ->dict[str, Any]:
        """Test stationarity of both series using ADF test."""
        results = {}
        for col in [target_col, predictor_col]:
            try:
                adf_result = adfuller(test_data[col])
                results[col] = {'adf_statistic': adf_result[0], 'p_value':
                    adf_result[1], 'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05, 'is_i1': not
                    adf_result[1] < 0.05}
            except Exception as e:
                logger.error(f"Stationarity test failed for {col}: {e}", exc_info=True)
                results[col] = {'error': str(e), 'is_stationary': False,
                    'is_i1': False}
        return results

    @staticmethod
    def _select_optimal_lag(test_data: pd.DataFrame, maxlag: int,
        lag_selection: str) ->int:
        """Select optimal lag using information criteria."""
        try:
            model = VAR(test_data)
            lag_results = model.select_order(maxlags=maxlag)
            if hasattr(lag_results, lag_selection):
                return int(lag_results.aic if lag_selection == 'aic' else
                    lag_results.bic if lag_selection == 'bic' else
                    lag_results.hqic)
            else:
                return int(lag_results.aic)
        except Exception as e:
            logger.error(f'Lag selection failed: {e}', exc_info=True)
            return 2

    @staticmethod
    def _run_granger_with_validation(test_data: pd.DataFrame, lag: int) ->dict[
        str, Any]:
        """Run Granger causality test with additional validation."""
        try:
            granger_result = grangercausalitytests(test_data, maxlag=lag,
                verbose=False)
            f_p_values = [result['ssr_ftest'][1] for result in granger_result]
            min_p_value = min(f_p_values)
            best_lag = f_p_values.index(min_p_value) + 1
            model = VAR(test_data)
            fitted_model = model.fit(lag)
            residuals = fitted_model.resid
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            return {'p_value': min_p_value, 'best_lag': best_lag,
                'all_p_values': f_p_values, 'residual_diagnostics': {
                'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[0],
                'residuals_autocorrelated': lb_test['lb_pvalue'].iloc[0] <
                0.05}, 'is_valid': min_p_value < 0.05 and lb_test[
                'lb_pvalue'].iloc[0] > 0.05}
        except Exception as e:
            logger.error(f'Granger validation test failed: {e}')
            return {'error': str(e), 'p_value': 1.0, 'is_valid': False}

    @staticmethod
    def _test_cointegration(test_data: pd.DataFrame) ->dict[str, Any]:
        """Test for cointegration between two I(1) series."""
        try:
            result = coint_johansen(test_data, det_order=0, k_ar_diff=1)
            return {'trace_statistic': result.lr1[0], 'critical_values':
                result.cvt[:, 0], 'is_cointegrated': result.lr1[0] > result
                .cvt[0, 0], 'eigenvalue_statistic': result.lr2[0],
                'eigenvectors': result.evec}
        except Exception as e:
            logger.error(f'Cointegration test failed: {e}', exc_info=True)
            return {'error': str(e), 'is_cointegrated': False}

    @staticmethod
    def _calculate_impulse_response(test_data: pd.DataFrame, lag: int,
        periods: int=10) ->dict[str, Any]:
        """Calculate impulse response functions."""
        try:
            model = VAR(test_data)
            fitted_model = model.fit(lag)
            irf = fitted_model.irf(periods)
            irf_data = irf.irfs
            return {'response_data': irf_data, 'cumulative_effect': irf.
                cum_effects, 'confidence_intervals': irf.cum_effects_ci}
        except Exception as e:
            logger.error(f'Impulse response calculation failed: {e}', exc_info=True)
            return {'error': str(e)}

    @staticmethod
    def _calculate_variance_decomposition(test_data: pd.DataFrame, lag: int,
        periods: int=10) ->dict[str, Any]:
        """Calculate forecast error variance decomposition."""
        try:
            model = VAR(test_data)
            fitted_model = model.fit(lag)
            fevd = fitted_model.fevd(periods)
            return {'decomposition': fevd.decomp, 'explained_variance':
                fevd.cumm_effects}
        except Exception as e:
            logger.error(f'Variance decomposition failed: {e}', exc_info=True)
            return {'error': str(e)}

    @staticmethod
    def _calculate_causality_strength(granger_results: dict[str, Any],
        stationarity: dict[str, Any], cointegration: dict[str, Any]) ->float:
        """Calculate comprehensive causality strength metric."""
        base_strength = 1 - granger_results.get('p_value', 1.0)
        stationarity_bonus = 0.0
        if stationarity.get('target', {}).get('is_stationary', False):
            stationarity_bonus += 0.1
        cointegration_bonus = 0.0
        if cointegration and cointegration.get('is_cointegrated', False):
            cointegration_bonus += 0.2
        residual_bonus = 0.0
        if not granger_results.get('residual_diagnostics', {}).get(
            'residuals_autocorrelated'):
            residual_bonus += 0.1
        total_strength = (base_strength + stationarity_bonus +
            cointegration_bonus + residual_bonus)
        return float(min(total_strength, 1.0))

    @staticmethod
    def _generate_causality_summary(causality_results: dict[str, Any]) ->dict[
        str, Any]:
        """Generate summary statistics for causality analysis."""
        if not causality_results:
            return {'error': 'No causality results to summarize'}
        results = {k: v for k, v in causality_results.items() if not k.
            startswith('_')}
        significant_count = sum(1 for r in results.values() if r.get(
            'is_significant', False))
        total_count = len(results)
        avg_strength = np.mean([r.get('causality_strength', 0) for r in
            results.values() if 'causality_strength' in r])
        return {'total_tests': total_count, 'significant_relationships':
            significant_count, 'significance_rate': significant_count /
            total_count if total_count > 0 else 0,
            'average_causality_strength': avg_strength,
            'strongest_predictor': max(results.keys(), key=lambda k:
            results[k].get('causality_strength', 0)) if results else None,
            'recommendation': 'Strong causal relationships found' if
            significant_count > 0 else 'No significant causal relationships'}

    @staticmethod
    def _validate_columns(df: pd.DataFrame, target_col: str, predictor_col: str
        ) ->bool:
        """Validate that required columns exist in DataFrame."""
        return target_col in df.columns and predictor_col in df.columns

    @staticmethod
    def _validate_data_length(test_data: pd.DataFrame, maxlag: int,
        predictor_col: str) ->bool:
        """Validate that sufficient data exists for Granger test."""
        min_required = maxlag + 10
        if len(test_data) < min_required:
            logger.warning(
                f"Insufficient data for Granger test on '{predictor_col}'. Required: {min_required}, Available: {len(test_data)}"
                )
            return False
        return True
