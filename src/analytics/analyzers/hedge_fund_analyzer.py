"""
Hedge Fund Analyzer
Comprehensive evaluation of model/strategy performance as a professional investment vehicle.
Includes risk-reward metrics, Fama-French factor exposures, and style drift detection.
"""
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Any, Optional
from datetime import datetime
from ..interfaces import IAnalyzer
from ..calculators.fama_french_factors import FamaFrenchFactors
from ..calculators.drawdown_calculator import DrawdownCalculator
from ..calculators.risk_reward_calculator import RiskRewardCalculator
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError
logger = ProjectLogger.get_logger(__name__)


class HedgeFundAnalyzer(IAnalyzer):
    """
    Advanced analysis module for evaluating quantitative strategies through a hedge fund lens.
    Evaluates statistical skill, benchmark relative performance, and potential style drift.
    """

    def __init__(self, factor_provider: Optional[FamaFrenchFactors]=None,
        **kwargs):
        """
        Initializes the HedgeFundAnalyzer with configurable thresholds.

        Args:
            factor_provider: Provider for Fama-French research factors.
            **kwargs: Configuration like risk_free_rate and style_thresholds.
        """
        self.factor_provider = factor_provider or FamaFrenchFactors()
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.02)
        self.periods_per_year = kwargs.get('periods_per_year', 252)
        self.style_thresholds = kwargs.get('style_thresholds', {
            'alpha_significance': 0.05})
        self.factor_models = {'carhart': ['MKT', 'SMB', 'HML', 'UMD'],
            'french_5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA']}
        logger.info(
            'HedgeFundAnalyzer initialized for institutional-grade evaluation.'
            )

    def analyze(self, data_map: Dict[str, Any], **kwargs) ->Dict[str, Any]:
        """
        Orchestrates full performance and risk decomposition for a return stream.

        Args:
            data_map: Dictionary containing 'returns' series and optional 'benchmark' series.
            **kwargs: Contextual overrides like 'historical_exposures'.
        """
        returns = data_map.get('returns')
        benchmark = data_map.get('benchmark')
        if returns is None or not isinstance(returns, pd.Series) or returns.empty:
            raise DataProcessingError('Returns data is missing or invalid.')

        try:
            performance = self.calculate_performance_metrics(returns, benchmark)
            factor_results = self.calculate_factor_exposures(returns)
            skill_analysis = self.analyze_manager_skill(returns,
                performance, factor_results)
            historical_exposures = kwargs.get('historical_exposures', [])
            drift_analysis = self.detect_style_drift(factor_results.get(
                'exposures', {}), historical_exposures)
            return {'performance': performance, 'factor_analysis':
                factor_results, 'skill_assessment': skill_analysis,
                'style_drift': drift_analysis, 'analysis_timestamp':
                datetime.now().isoformat()}
        except Exception as e:
            logger.error(f'Execution failure in HedgeFundAnalyzer: {e}', exc_info=True)
            raise DataProcessingError(f'Execution failure in HedgeFundAnalyzer: {e}') from e

    def calculate_performance_metrics(self, returns: pd.Series, benchmark:
        Optional[pd.Series]=None) ->Dict[str, float]:
        """Calculates institutional risk-reward metrics using centralized calculators."""
        metrics = {}

        metrics['annual_return'] = returns.mean() * self.periods_per_year
        metrics['annual_volatility'] = returns.std() * np.sqrt(self.
            periods_per_year)
        metrics['sharpe_ratio'
            ] = RiskRewardCalculator.calculate_sharpe_ratio(returns,
            self.risk_free_rate, self.periods_per_year)
        metrics['sortino_ratio'
            ] = RiskRewardCalculator.calculate_sortino_ratio(returns,
            self.risk_free_rate, self.periods_per_year)
        drawdown_series = (DrawdownCalculator.
            calculate_max_drawdown_from_returns(returns))
        metrics['max_drawdown'] = float(drawdown_series.min()
            ) if not drawdown_series.empty else 0.0
        var_cvar_results = RiskRewardCalculator.calculate_var_cvar(returns,
            confidence_level=0.95)
        metrics['var_95'] = var_cvar_results['var']
        metrics['cvar_95'] = var_cvar_results['cvar']
        if benchmark is not None:
            metrics['beta'] = RiskRewardCalculator.calculate_beta(returns,
                benchmark)
            metrics['treynor_ratio'
                ] = RiskRewardCalculator.calculate_treynor_ratio(returns,
                benchmark, self.risk_free_rate, self.periods_per_year)
            metrics['information_ratio'
                ] = RiskRewardCalculator.calculate_information_ratio(
                returns, benchmark, self.periods_per_year)
        return {k: (v if not pd.isna(v) else None) for k, v in metrics.
            items()}

    def calculate_factor_exposures(self, returns: pd.Series, model_name:
        str='carhart') ->Dict[str, Any]:
        """Estimates portfolio sensitivity to systematic risk factors."""
        min_idx = returns.index.min()
        max_idx = returns.index.max()
        if not isinstance(min_idx, (pd.Timestamp, datetime)):
            try:
                min_idx = pd.to_datetime(min_idx)
            except (ValueError, TypeError):
                min_idx = pd.Timestamp.now() - pd.Timedelta(days=365)
        if not isinstance(max_idx, (pd.Timestamp, datetime)):
            try:
                max_idx = pd.to_datetime(max_idx)
            except (ValueError, TypeError):
                max_idx = pd.Timestamp.now()

        start_date = min_idx.strftime('%Y-%m-%d')
        end_date = max_idx.strftime('%Y-%m-%d')

        factors_df = self.factor_provider.get_factors(start_date, end_date)

        if factors_df is None or factors_df.empty:
            raise DataProcessingError(f'Factor dataset empty or unavailable for {start_date} - {end_date}')

        model_factors = self.factor_models.get(model_name, self.
            factor_models['carhart'])
        X = factors_df[[f for f in model_factors if f in factors_df.
            columns]]
        common_idx = returns.index.intersection(X.index)
        if len(common_idx) < 20:
            raise DataProcessingError(f'Insufficient temporal overlap for factor regression: {len(common_idx)} points.')

        y_sub = returns.loc[common_idx]
        x_sub = sm.add_constant(X.loc[common_idx])
        model = sm.OLS(y_sub, x_sub).fit()
        return {'exposures': model.params.to_dict(), 'p_values': model.
            pvalues.to_dict(), 'r_squared': model.rsquared,
            'adjust_r_squared': model.rsquared_adj}

    def detect_style_drift(self, current_exposures: Dict[str, float],
        historical_exposures: List[Dict[str, float]]) ->Dict[str, Any]:
        """Identifies statistical deviations from historical stylistic baselines."""
        if not historical_exposures or not current_exposures:
            return {'drift_detected': False, 'message':
                'Insufficient history for drift detection.'}
        drifts = {}
        for factor, current_val in current_exposures.items():
            if factor == 'const':
                continue
            hist_vals = [h.get(factor, 0) for h in historical_exposures if 
                h and factor in h]
            if len(hist_vals) > 5:
                mean_hist = np.mean(hist_vals)
                std_hist = np.std(hist_vals)
                z_score = abs(current_val - mean_hist) / (std_hist if 
                    std_hist > 0 else 0.01)
                drifts[factor] = {'z_score': z_score, 'significant': 
                    z_score > 2.0}
        drift_detected = any(d.get('significant', False) for d in drifts.
            values())
        return {'drift_detected': drift_detected, 'factor_drifts': drifts}

    def analyze_manager_skill(self, returns: pd.Series, performance: Dict,
        factors: Dict) ->Dict[str, Any]:
        """Categorizes Alpha generation as either structural skill or coincidental beta."""
        alpha = factors.get('exposures', {}).get('const', 0
            ) * self.periods_per_year
        alpha_p_value = factors.get('p_values', {}).get('const', 1.0)
        is_alpha_significant = alpha_p_value < self.style_thresholds[
            'alpha_significance']
        score_components = []
        if alpha > 0 and is_alpha_significant:
            score_components.append(1.0)
        elif alpha > 0:
            score_components.append(0.5)
        else:
            score_components.append(0.0)
        sharpe = performance.get('sharpe_ratio')
        if sharpe is not None:
            score_components.append(min(max(float(sharpe) / 2.0, 0), 1.0))
        win_rate = (returns > 0).mean() if not returns.empty else 0
        score_components.append(win_rate)
        final_score = np.mean(score_components) if score_components else 0.0
        return {'alpha_annualized': float(alpha), 'is_alpha_significant':
            bool(is_alpha_significant), 'p_value_alpha': float(
            alpha_p_value), 'skill_score': float(final_score),
            'manager_rating': self._get_manager_rating(final_score)}

    def _get_manager_rating(self, final_score: float) ->str:
        """Determine manager rating based on final score."""
        if final_score > 0.8:
            return 'Exceptional'
        elif final_score > 0.6:
            return 'Commendable'
        else:
            return 'Standard'

