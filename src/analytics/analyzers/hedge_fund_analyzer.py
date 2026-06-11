# src/analytics/analyzers/hedge_fund_analyzer.py
"""
Hedge Fund Analyzer
Comprehensive evaluation of model/strategy performance as a professional investment vehicle.
Includes risk-reward metrics, Fama-French factor exposures, and style drift detection.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator, TradeConfig
from src.analytics.interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger

from ..calculators.fama_french_factors import FamaFrenchFactors

logger = ProjectLogger.get_logger(__name__)

class HedgeFundAnalyzer(IAnalyzer):
    """
    Advanced analysis module for evaluating quantitative strategies through a hedge fund lens.
    Evaluates statistical skill, benchmark relative performance, and potential style drift.
    """

    def __init__(self, factor_provider: FamaFrenchFactors | None = None, **kwargs):
        """
        Initializes the HedgeFundAnalyzer with configurable thresholds.

        Args:
            factor_provider: Provider for Fama-French research factors.
            **kwargs: Configuration like risk_free_rate and style_thresholds.
        """
        self.factor_provider = factor_provider or FamaFrenchFactors()
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.02)  # Annualized baseline
        self.periods_per_year = kwargs.get('periods_per_year', 252)
        self.style_thresholds = kwargs.get('style_thresholds', {
            'alpha_significance': 0.05
        })
        self.factor_models = {
            'carhart': ['MKT', 'SMB', 'HML', 'UMD'],
            'french_5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        }
        logger.info("HedgeFundAnalyzer initialized for institutional-grade evaluation.")

    def analyze(self, data: Any, **kwargs) -> dict[str, Any]:
        """
        Orchestrates full performance and risk decomposition for a return stream.

        Args:
            data: Dictionary containing 'returns' series and optional 'benchmark' series.
            **kwargs: Contextual overrides like 'historical_exposures'.
        """
        data_map = data if isinstance(data, dict) else {}
        returns = data_map.get('returns')
        benchmark = data_map.get('benchmark')

        if returns is None or not isinstance(returns, pd.Series) or returns.empty:
            logger.error("HedgeFundAnalyzer received invalid return series.")
            return {"error": "Invalid or missing returns data"}

        try:
            performance_metrics = self.calculate_performance_metrics(returns, benchmark)
            factor_exposures = self.calculate_factor_exposures(returns)
            style_drift = self.detect_style_drift(
                factor_exposures, 
                list(data_map.get('historical_exposures', [])) if isinstance(data_map.get('historical_exposures'), list) else []
            )

            return {
                'performance': performance_metrics,
                'factor_exposures': factor_exposures,
                'style_drift': style_drift,
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'error': f'Analysis failed: {str(e)}'}

    def calculate_performance_metrics(self, returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
        """Calculates institutional risk-reward metrics using centralized calculators."""
        metrics = {}
        try:
            metrics['annual_return'] = returns.mean() * self.periods_per_year
            metrics['annual_volatility'] = returns.std() * np.sqrt(self.periods_per_year)

            # Shared Financial Logic
            metrics['sharpe_ratio'] = RiskRewardCalculator.calculate_sharpe_ratio(
                returns, TradeConfig(risk_free_rate=self.risk_free_rate, periods_per_year=self.periods_per_year)
            )
            metrics['sortino_ratio'] = RiskRewardCalculator.calculate_sortino_ratio(
                returns, TradeConfig(risk_free_rate=self.risk_free_rate, periods_per_year=self.periods_per_year)
            )

            drawdown_series = DrawdownCalculator.calculate_max_drawdown_from_returns(returns)
            metrics['max_drawdown'] = float(drawdown_series.min()) if not drawdown_series.empty else 0.0

            # Volatility-adjusted risk metrics
            var_cvar_results = RiskRewardCalculator.calculate_var_cvar(
                returns, TradeConfig(confidence_level=0.95)
            )
            metrics['var_95'] = var_cvar_results['var']
            metrics['cvar_95'] = var_cvar_results['cvar']

            if benchmark is not None:
                metrics['beta'] = RiskRewardCalculator.calculate_beta(returns, benchmark)
                metrics['treynor_ratio'] = RiskRewardCalculator.calculate_treynor_ratio(
                    returns, benchmark, TradeConfig(risk_free_rate=self.risk_free_rate, periods_per_year=self.periods_per_year)
                )
                metrics['information_ratio'] = RiskRewardCalculator.calculate_information_ratio(
                    returns, benchmark, TradeConfig(periods_per_year=self.periods_per_year)
                )

            # Sanitize floating point results for JSON serialization
            sanitized = {}
            for k, v in metrics.items():
                if v is not None and not pd.isna(v):
                    sanitized[k] = float(v)
                else:
                    sanitized[k] = 0.0
            return sanitized

        except Exception as e:
            logger.warning(f"Metrical calculation partial failure: {e}")
            return {}

    def calculate_factor_exposures(self, returns: pd.Series, model_name: str = 'carhart') -> dict[str, Any]:
        """Estimates portfolio sensitivity to systematic risk factors."""
        try:
            # Datetime normalization and boundary validation
            min_idx = returns.index.min()
            max_idx = returns.index.max()

            if not isinstance(min_idx, (pd.Timestamp, datetime)):
                try:
                    min_idx = pd.to_datetime(min_idx)
                except (ValueError, TypeError):
                    min_idx = pd.Timestamp.now() - pd.Timedelta(days=365)
                    logger.debug(f"Falling back to default start_date for factor audit: {min_idx}")

            if not isinstance(max_idx, (pd.Timestamp, datetime)):
                try:
                    max_idx = pd.to_datetime(max_idx)
                except (ValueError, TypeError):
                    max_idx = pd.Timestamp.now()
                    logger.debug(f"Falling back to default end_date for factor audit: {max_idx}")

            start_date = min_idx.strftime('%Y-%m-%d')
            end_date = max_idx.strftime('%Y-%m-%d')

            # Fetch research factors from remote/local providers
            try:
                factors_df = self.factor_provider.get_factors(start_date, end_date)
            except Exception as e:
                logger.warning(f"Factor retrieval failed for period {start_date} to {end_date}: {e}")
                return {"error": "Factor source unavailable."}

            if factors_df is None or factors_df.empty:
                logger.warning(f"No usable factor data found for range: {start_date} - {end_date}")
                return {"error": "Factor dataset empty for specified range."}

            model_factors = self.factor_models.get(model_name, self.factor_models['carhart'])
            X = factors_df[[f for f in model_factors if f in factors_df.columns]]

            # Ensure temporal alignment across disparate sources
            common_idx = returns.index.intersection(X.index)
            if len(common_idx) < 20:
                logger.warning(f"Insufficient temporal overlap for factor regression: {len(common_idx)} points.")
                return {"error": "Insufficient overlapping data points."}

            y_sub = returns.loc[common_idx]
            x_sub = sm.add_constant(X.loc[common_idx])

            # Ordinary Least Squares Regression for exposure estimation
            model = sm.OLS(y_sub, x_sub).fit()

            return {
                'exposures': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'r_squared': model.rsquared,
                'adjust_r_squared': model.rsquared_adj
            }
        except Exception as e:
            logger.error(f"Factor exposure analysis exception: {e}")
            return {}

    def detect_style_drift(self, current_exposures: dict[str, float], historical_exposures: list[dict[str, float]]) -> dict[str, Any]:
        """Identifies statistical deviations from historical stylistic baselines."""
        if not historical_exposures or not current_exposures:
            return {'drift_detected': False, 'message': 'Insufficient history for drift detection.'}

        drifts = {}
        for factor, current_val in current_exposures.items():
            if factor == 'const': continue
            hist_vals = [h.get(factor, 0) for h in historical_exposures if h and factor in h]
            if len(hist_vals) > 5:
                mean_hist = np.mean(hist_vals)
                std_hist = np.std(hist_vals)
                # Z-score represents distance from mean in standard deviations
                z_score = abs(current_val - mean_hist) / (std_hist if std_hist > 0 else 0.01)
                drifts[factor] = {'z_score': z_score, 'significant': z_score > 2.0}

        drift_detected = any(d.get('significant', False) for d in drifts.values())
        return {'drift_detected': drift_detected, 'factor_drifts': drifts}

    def analyze_manager_skill(self, returns: pd.Series, performance: dict, factors: dict) -> dict[str, Any]:
        """Categorizes Alpha generation as either structural skill or coincidental beta."""
        alpha = factors.get('exposures', {}).get('const', 0) * self.periods_per_year
        alpha_p_value = factors.get('p_values', {}).get('const', 1.0)
        is_alpha_significant = alpha_p_value < self.style_thresholds['alpha_significance']

        # Skill Score Calculation (0.0 - 1.0)
        score_components = []

        # 1. Alpha Quality
        if alpha > 0 and is_alpha_significant: score_components.append(1.0)
        elif alpha > 0: score_components.append(0.5)
        else: score_components.append(0.0)

        # 2. Risk Adjustment (Sharpe)
        sharpe = performance.get('sharpe_ratio')
        if sharpe is not None: score_components.append(min(max(float(sharpe) / 2.0, 0), 1.0))

        # 3. Decision Consistency
        win_rate = (returns > 0).mean() if not returns.empty else 0
        score_components.append(win_rate)

        final_score = np.mean(score_components) if score_components else 0.0

        return {
            'alpha_annualized': float(alpha),
            'is_alpha_significant': bool(is_alpha_significant),
            'p_value_alpha': float(alpha_p_value),
            'skill_score': float(final_score),
            'manager_rating': self._get_manager_rating(final_score)
        }

    def _get_manager_rating(self, final_score: float) -> str:
        """Determine manager rating based on final score."""
        if final_score > 0.8:
            return 'Exceptional'
        elif final_score > 0.6:
            return 'Commendable'
        else:
            return 'Standard'
