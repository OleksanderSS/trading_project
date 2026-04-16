import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..interfaces import IAnalyzer
from ..calculators.fama_french_factors import FamaFrenchFactors
from ..calculators.drawdown_calculator import DrawdownCalculator
from ..calculators.risk_reward_calculator import RiskRewardCalculator

logger = logging.getLogger(__name__)

class HedgeFundAnalyzer(IAnalyzer):
    """
    Comprehensive analysis module for evaluating model/strategy performance 
    as a hedge fund, including risk metrics, factor exposures, skill, and style drift.
    """
    
    def __init__(self, factor_provider: Optional[FamaFrenchFactors] = None, **kwargs):
        """
        Initializes the HedgeFundAnalyzer.
        
        Args:
            factor_provider: Instance of FamaFrenchFactors for exposure analysis.
            **kwargs: Configuration parameters like style_thresholds.
        """
        self.factor_provider = factor_provider or FamaFrenchFactors()
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.02) # Annualized
        self.periods_per_year = kwargs.get('periods_per_year', 252)
        self.style_thresholds = kwargs.get('style_thresholds', {
            'alpha_significance': 0.05
        })
        self.factor_models = {
            'carhart': ['MKT', 'SMB', 'HML', 'UMD'],
            'french_5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        }
        logger.info("HedgeFundAnalyzer initialized.")

    def analyze(self, data_map: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main entry point for UnifiedAnalyticsEngine.
        
        Args:
            data_map: Dict containing 'returns' (pd.Series) and optionally 'benchmark' (pd.Series).
            **kwargs: Optional 'historical_exposures' (List[Dict]) for drift detection.
        """
        returns = data_map.get('returns')
        benchmark = data_map.get('benchmark')

        if returns is None or not isinstance(returns, pd.Series) or returns.empty:
            logger.error("HedgeFundAnalyzer: Missing or invalid 'returns' data.")
            return {"error": "Invalid returns data"}

        try:
            performance = self.calculate_performance_metrics(returns, benchmark)
            factor_results = self.calculate_factor_exposures(returns)
            skill_analysis = self.analyze_manager_skill(returns, performance, factor_results)
            
            historical_exposures = kwargs.get('historical_exposures', [])
            drift_analysis = self.detect_style_drift(factor_results.get('exposures', {}), historical_exposures)

            return {
                "performance": performance,
                "factor_analysis": factor_results,
                "skill_assessment": skill_analysis,
                "style_drift": drift_analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"HedgeFundAnalyzer error: {e}", exc_info=True)
            return {"error": str(e)}

    def calculate_performance_metrics(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculates key performance and risk metrics using centralized calculators.
        """
        metrics = {}
        try:
            metrics['annual_return'] = returns.mean() * self.periods_per_year
            metrics['annual_volatility'] = returns.std() * np.sqrt(self.periods_per_year)
            
            # --- Use centralized calculators for all risk/reward metrics ---
            metrics['sharpe_ratio'] = RiskRewardCalculator.calculate_sharpe_ratio(
                returns, self.risk_free_rate, self.periods_per_year
            )
            metrics['sortino_ratio'] = RiskRewardCalculator.calculate_sortino_ratio(
                returns, self.risk_free_rate, self.periods_per_year
            )
            
            drawdown_series = DrawdownCalculator.calculate_max_drawdown_from_returns(returns)
            metrics['max_drawdown'] = drawdown_series.min() if not drawdown_series.empty else 0

            # Use the new centralized VaR/CVaR calculator
            var_cvar_results = RiskRewardCalculator.calculate_var_cvar(returns, confidence_level=0.95)
            metrics['var_95'] = var_cvar_results['var']
            metrics['cvar_95'] = var_cvar_results['cvar']
            
            if benchmark is not None:
                metrics['beta'] = RiskRewardCalculator.calculate_beta(returns, benchmark)
                metrics['treynor_ratio'] = RiskRewardCalculator.calculate_treynor_ratio(
                    returns, benchmark, self.risk_free_rate, self.periods_per_year
                )
                # Use the new centralized Information Ratio calculator
                metrics['information_ratio'] = RiskRewardCalculator.calculate_information_ratio(
                    returns, benchmark, self.periods_per_year
                )

            # Clean NaNs before returning
            return {k: (v if not pd.isna(v) else None) for k, v in metrics.items()}

        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}", exc_info=True)
            return {}

    def calculate_factor_exposures(self, returns: pd.Series, model_name: str = 'carhart') -> Dict[str, Any]:
        """Calculates exposure to Fama-French factors using statsmodels."""
        try:
            # ✅ FIX: Convert to datetime if needed (handles numpy.int64 case)
            min_idx = returns.index.min()
            max_idx = returns.index.max()
            
            # Convert to datetime if not already
            if not isinstance(min_idx, (pd.Timestamp, datetime)):
                try:
                    min_idx = pd.to_datetime(min_idx)
                except (ValueError, TypeError):
                    # Якщо не вдається конвертувати, використовуємо поточну дату мінус 1 рік
                    min_idx = pd.Timestamp.now() - pd.Timedelta(days=365)
                    logger.warning(f"Could not convert min_idx to datetime, using fallback: {min_idx}")
            
            if not isinstance(max_idx, (pd.Timestamp, datetime)):
                try:
                    max_idx = pd.to_datetime(max_idx)
                except (ValueError, TypeError):
                    # Якщо не вдається конвертувати, використовуємо поточну дату
                    max_idx = pd.Timestamp.now()
                    logger.warning(f"Could not convert max_idx to datetime, using fallback: {max_idx}")
            
            start_date = min_idx.strftime('%Y-%m-%d')
            end_date = max_idx.strftime('%Y-%m-%d')
            
            # ✅ FIX: Додаємо try-except для завантаження факторів
            try:
                factors_df = self.factor_provider.get_factors(start_date, end_date)
            except Exception as e:
                logger.warning(f"⚠️ Не вдалося завантажити Fama-French фактори: {e}")
                return {"error": "Factor data not available for the given date range."}
            
            if factors_df is None or factors_df.empty:
                logger.warning(f"⚠️ Fama-French фактори порожні для {start_date} - {end_date}")
                return {"error": "Factor data not available for the given date range."}

            model_factors = self.factor_models.get(model_name, self.factor_models['carhart'])
            X = factors_df[[f for f in model_factors if f in factors_df.columns]]
            
            common_idx = returns.index.intersection(X.index)
            if len(common_idx) < 20:
                return {"error": "Insufficient overlapping data for factor analysis."}
                
            y_sub = returns.loc[common_idx]
            X_sub = sm.add_constant(X.loc[common_idx])
            
            model = sm.OLS(y_sub, X_sub).fit()
            
            return {
                'exposures': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'r_squared': model.rsquared
            }
        except Exception as e:
            logger.warning(f"Factor exposure calculation failed: {e}")
            return {}

    def detect_style_drift(self, current_exposures: Dict[str, float], historical_exposures: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detects if current investment style deviates significantly from history."""
        if not historical_exposures or not current_exposures:
            return {'drift_detected': False}
            
        drifts = {}
        for factor, current_val in current_exposures.items():
            if factor == 'const': continue
            hist_vals = [h.get(factor, 0) for h in historical_exposures if h and factor in h]
            if len(hist_vals) > 5:
                mean_hist = np.mean(hist_vals)
                std_hist = np.std(hist_vals)
                z_score = abs(current_val - mean_hist) / (std_hist if std_hist > 0 else 0.01)
                drifts[factor] = {'z_score': z_score, 'significant': z_score > 2.0}
        
        drift_detected = any(d.get('significant', False) for d in drifts.values())
        return {'drift_detected': drift_detected, 'factor_drifts': drifts}

    def analyze_manager_skill(self, returns: pd.Series, performance: Dict, factors: Dict) -> Dict[str, Any]:
        """Evaluates skill based on Alpha significance and consistency."""
        alpha = factors.get('exposures', {}).get('const', 0) * self.periods_per_year
        alpha_p_value = factors.get('p_values', {}).get('const', 1.0)
        is_alpha_significant = alpha_p_value < self.style_thresholds['alpha_significance']
        
        score_components = []
        if alpha > 0 and is_alpha_significant: score_components.append(1.0)
        elif alpha > 0: score_components.append(0.5)
        
        sharpe = performance.get('sharpe_ratio')
        if sharpe is not None: score_components.append(min(max(sharpe / 2.0, 0), 1.0))
        
        win_rate = (returns > 0).mean() if not returns.empty else 0
        score_components.append(win_rate)
        
        final_score = np.mean(score_components) if score_components else 0
        
        return {
            'alpha_annualized': alpha,
            'is_alpha_significant': is_alpha_significant,
            'p_value_alpha': alpha_p_value,
            'skill_score': final_score,
            'rating': 'Exceptional' if final_score > 0.8 else ('Good' if final_score > 0.6 else 'Average')
        }
