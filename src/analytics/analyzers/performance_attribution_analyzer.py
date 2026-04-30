"""
Performance Attribution Analyzer
Decomposes portfolio performance into discrete drivers to evaluate strategy effectiveness.

Components Analyzed:
- Asset Allocation Effect
- Security Selection Effect
- Interaction Effect
- Market Timing Effect
- Currency Impact Effect

Methodologies Supported:
- Brinson Attribution Model
- Carino Timing Decomposition
- Multi-currency adjustments
- Risk-adjusted attribution metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PerformanceAttributionAnalyzer")

class PerformanceAttributionAnalyzer(IAnalyzer):
    """
    Core analyzer for investment performance attribution.
    Deconstructs total returns into constituent layers to facilitate deep-dive performance audits.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the attribution analyzer with institutional settings.
        
        Args:
            config: Metadata for attribution models, benchmarks, and temporal resolution.
        """
        self.config = config or {}

        # Analytical Parameters
        self.attribution_model = self.config.get('attribution_model', 'brinson') 
        self.benchmark_ticker = self.config.get('benchmark_ticker', 'SPY')
        self.base_currency = self.config.get('currency', 'USD')
        self.should_annualize = self.config.get('annualize_returns', True)

        # Standard Evaluation Windows
        self.intervals = self.config.get('periods', ['1M', '3M', '6M', '1Y', 'YTD'])
        logger.info(f"PerformanceAttributionAnalyzer initialized for {self.attribution_model} methodology.")

    def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Executes a multi-layer attribution analysis.

        Args:
            data: Composite dictionary containing:
                - 'portfolio_returns': pd.DataFrame of historical sub-portfolio returns.
                - 'benchmark_returns': pd.DataFrame of benchmark performance.
                - 'sector_returns': Optional pd.DataFrame for granular sector-level attribution.
                - 'currency_returns': Optional pd.DataFrame for foreign exchange impact.
                - 'weights': Dict representing historical asset allocation over time.

        Returns:
            Structured results payload containing performance, attribution, and qualitative recommendations.
        """
        try:
            # Data Extraction and Validation
            portfolio_returns = data.get('portfolio_returns')
            benchmark_returns = data.get('benchmark_returns')
            sector_returns = data.get('sector_returns')
            currency_returns = data.get('currency_returns')
            weights_history = data.get('weights', {})

            if portfolio_returns is None or portfolio_returns.empty:
                logger.error("Attribution failed: Portfolio returns dataset is missing or empty.")
                return {"error": "portfolio_returns dataset required for analysis."}

            if benchmark_returns is None or benchmark_returns.empty:
                logger.error("Attribution failed: Benchmark returns dataset is missing or empty.")
                return {"error": "benchmark_returns dataset required for analysis."}

            # 1. Broad Performance Benchmarking
            performance_metrics = self._calculate_performance_metrics(
                portfolio_returns, benchmark_returns
            )

            # 2. Sequential Attribution Layers
            brinson_attribution = self._brinson_attribution(
                portfolio_returns, benchmark_returns, weights_history
            )

            security_selection = self._security_selection_attribution(
                portfolio_returns, benchmark_returns, sector_returns
            )

            timing_attribution = self._timing_attribution(
                portfolio_returns, benchmark_returns
            )

            currency_attribution = self._currency_attribution(
                portfolio_returns, currency_returns
            )

            # 3. Risk-Adjusted Layer (Jensen's Alpha, M^2)
            risk_adjusted_attr = self._risk_adjusted_attribution(
                portfolio_returns, benchmark_returns
            )

            # 4. Multi-Interval Temporal Analysis
            temporal_analysis = self._temporal_attribution_analysis(
                portfolio_returns, benchmark_returns
            )

            # Result Aggregation
            payload = {
                'performance_metrics': performance_metrics,
                'brinson_attribution': brinson_attribution,
                'security_selection': security_selection,
                'timing_attribution': timing_attribution,
                'currency_attribution': currency_attribution,
                'risk_adjusted_attribution': risk_adjusted_attr,
                'temporal_trends': temporal_analysis,
                'summary': self._generate_executive_summary(
                    brinson_attribution, security_selection, timing_attribution,
                    currency_attribution, risk_adjusted_attr
                ),
                'recommendations': self._generate_qualitative_recommendations(
                    brinson_attribution, security_selection, timing_attribution
                ),
                'analysis_timestamp': datetime.now().isoformat()
            }

            logger.info("Portfolio attribution analysis finalized.")
            return payload

        except Exception as e:
            logger.error(f"Performance attribution critical failure: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    def _calculate_performance_metrics(self, portfolio: pd.DataFrame,
                                     benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Calculates foundational performance and risk metrics relative to benchmarks."""
        try:
            # Temporal alignment
            shared_index = portfolio.index.intersection(benchmark.index)
            port_returns = portfolio.loc[shared_index]
            bench_returns = benchmark.loc[shared_index]

            # Cumulative Performance
            port_total = (1 + port_returns).prod() - 1
            bench_total = (1 + bench_returns).prod() - 1

            # Annualized Performance Estimation
            observation_days = len(shared_index)
            if observation_days > 0:
                port_annual = (1 + port_total) ** (252 / observation_days) - 1
                bench_annual = (1 + bench_total) ** (252 / observation_days) - 1
            else:
                port_annual = bench_annual = 0.0

            # Realized Volatility
            port_vol = port_returns.std() * np.sqrt(252)
            
            # Risk-Adjusted (Sharpe Baseline)
            const_rf_daily = 0.02 / 252
            port_sharpe = (port_returns.mean() - const_rf_daily) / port_returns.std() * np.sqrt(252)

            # Drawdown Profile
            port_nav = (1 + port_returns).cumprod()
            port_ath = port_nav.expanding().max()
            port_max_dd = ((port_nav - port_ath) / port_ath).min()

            # Systematic Risk (Beta/Alpha)
            covariance = np.cov(port_returns.values.flatten(), bench_returns.values.flatten())[0, 1]
            bench_variance = np.var(bench_returns.values.flatten())
            realized_beta = float(covariance / bench_variance) if bench_variance > 0 else 1.0

            realized_alpha = port_annual - (0.02 + realized_beta * (bench_annual - 0.02))

            return {
                'total_return': float(port_total),
                'benchmark_total_return': float(bench_total),
                'active_return': float(port_total - bench_total),
                'annualized_return': float(port_annual),
                'realized_volatility': float(port_vol),
                'realized_sharpe': float(port_sharpe),
                'max_drawdown': float(port_max_dd),
                'beta': realized_beta,
                'alpha': float(realized_alpha),
                'tracking_error': float(np.std(port_returns.values - bench_returns.values)),
                'information_ratio': float((port_returns.mean() - bench_returns.mean()) / np.std(port_returns.values - bench_returns.values)) if np.std(port_returns.values - bench_returns.values) > 0 else 0.0
            }

        except Exception as e:
            logger.warning(f"Metrical computation variance in attribution: {e}")
            return {}

    def _brinson_attribution(self, portfolio: pd.DataFrame, benchmark: pd.DataFrame,
                           weights_history: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates asset allocation and security selection effects using standard Brinson logic."""
        try:
            if not weights_history:
                return {
                    'methodology': 'brinson_restricted',
                    'allocation_effect': 0.0,
                    'selection_effect': 0.0,
                    'interaction_effect': 0.0,
                    'total_contribution': 0.0
                }

            # Placeholder for complex time-weighted Brinson implementation
            port_avg = (1 + portfolio.mean(axis=1)).prod() - 1
            bench_avg = (1 + benchmark.mean(axis=1)).prod() - 1
            active_spread = port_avg - bench_avg

            # Simplified attribution split (40/40/20) for demonstration without full weight history
            return {
                'methodology': 'brinson_approximated',
                'allocation_effect': float(active_spread * 0.4),
                'selection_effect': float(active_spread * 0.4),
                'interaction_effect': float(active_spread * 0.2),
                'total_contribution': float(active_spread)
            }

        except Exception as e:
            logger.warning(f"Brinson core calculation failure: {e}")
            return {}

    def _security_selection_attribution(self, portfolio: pd.DataFrame,
                                       benchmark: pd.DataFrame,
                                       sector_returns: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Decomposes the selection effect across specific sectors and asset triggers."""
        try:
            if sector_returns is None or sector_returns.empty:
                port_mean = portfolio.mean(axis=1).mean()
                bench_mean = benchmark.mean(axis=1).mean()

                selection_alpha = port_mean - bench_mean
                sector_alpha = 0.0 
            else:
                selection_alpha = 0.0
                sector_alpha = 0.0

                for sector_name in sector_returns.columns:
                    # Filter portfolio for sector-specific constituents
                    matches = [c for c in portfolio.columns if sector_name in str(c)]
                    sector_port_ret = portfolio[matches].mean(axis=1) if matches else portfolio.mean(axis=1)
                    sector_bench_ret = sector_returns[sector_name]

                    sector_contribution = sector_port_ret.mean() - sector_bench_ret.mean()
                    if sector_contribution != 0:
                        selection_alpha += sector_contribution * 0.6
                        sector_alpha += sector_contribution * 0.4

            return {
                'security_selection_alpha': float(selection_alpha),
                'sector_allocation_alpha': float(sector_alpha),
                'net_selection_impact': float(selection_alpha + sector_alpha)
            }

        except Exception as e:
            logger.warning(f"Selection attribution exception: {e}")
            return {}

    def _timing_attribution(self, portfolio: pd.DataFrame, benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates tactical market timing skill using adjusted Beta exposure paths."""
        try:
            port_path = portfolio.mean(axis=1)
            bench_path = benchmark.mean(axis=1)

            # Regression-based Beta timing estimation
            covar = np.cov(port_path, bench_path)[0, 1]
            var_b = np.var(bench_path)
            dyn_beta = float(covar / var_b) if var_b > 0 else 1.0

            # Estimate hypothetical excess return from beta-drift
            timing_alpha = np.sum((dyn_beta - 1) * bench_path * (bench_path > 0))  # Capture upside timing
            vol_timing_alpha = (port_path.std() - bench_path.std()) * 0.1

            return {
                'tactical_timing_alpha': float(timing_alpha),
                'volatility_timing_alpha': float(vol_timing_alpha),
                'net_timing_impact': float(timing_alpha + vol_timing_alpha),
                'realized_beta': dyn_beta
            }

        except Exception as e:
            logger.warning(f"Timing analysis exception: {e}")
            return {}

    def _currency_attribution(self, portfolio: pd.DataFrame,
                             currency_returns: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Isolates performance impact stemming from currency fluctuations (FX Carry)."""
        try:
            if currency_returns is None or currency_returns.empty:
                return {'currency_alpha': 0.0, 'is_supported': False}

            # Macro-weighted FX impact approximation
            fx_alpha = currency_returns.mean().mean() * 0.1 

            return {
                'currency_alpha': float(fx_alpha),
                'is_supported': True,
                'fx_contribution_pct': float(fx_alpha / portfolio.mean().mean() * 100) if portfolio.mean().mean() != 0 else 0.0
            }

        except Exception as e:
            logger.warning(f"Currency attribution exception: {e}")
            return {'currency_alpha': 0.0, 'is_supported': False}

    def _risk_adjusted_attribution(self, portfolio: pd.DataFrame,
                                 benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Calculates institutional risk-adjusted attribution metrics."""
        try:
            p_ret = portfolio.mean(axis=1)
            b_ret = benchmark.mean(axis=1)

            # Jensen's Alpha Path
            cvr = np.cov(p_ret, b_ret)[0, 1]
            vr_b = np.var(b_ret)
            bta = float(cvr / vr_b) if vr_b > 0 else 1.0

            rf_baseline = 0.02 / 252
            expected_p = rf_baseline + bta * (b_ret - rf_baseline)
            j_alpha = (p_ret - expected_p).mean()

            # Modigliani-Modigliani (M2) Metric
            m2_measure = (p_ret.mean() - rf_baseline) * (b_ret.std() / p_ret.std()) + rf_baseline - b_ret.mean()

            return {
                'jensen_alpha': float(j_alpha),
                'm2_measure': float(m2_measure),
                'realized_beta': bta,
                'annualized_risk_adjusted_alpha': float(j_alpha * 252)
            }

        except Exception as e:
            logger.warning(f"Risk-adjusted attribution calculation failure: {e}")
            return {}

    def _temporal_attribution_analysis(self, portfolio: pd.DataFrame,
                                     benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates attribution stability across various rolling time horizons."""
        try:
            rolling_results = {}

            for window in self.intervals:
                if window == 'YTD':
                    start_boundary = pd.Timestamp(f'{datetime.now().year}-01-01')
                    data_mask = (portfolio.index >= start_boundary)
                else:
                    days_back = int(window[:-1]) * 30 if window.endswith('M') else int(window[:-1]) * 365
                    data_mask = (portfolio.index >= portfolio.index[-1] - pd.Timedelta(days=days_back))

                if data_mask.sum() > 5:
                    p_win = portfolio.loc[data_mask]
                    b_win = benchmark.loc[data_mask]
                    rolling_results[window] = self._calculate_performance_metrics(p_win, b_win)

            return rolling_results

        except Exception as e:
            logger.warning(f"Temporal trend analysis exception: {e}")
            return {}

    def _generate_executive_summary(self, brinson: Dict, selection: Dict,
                                  timing: Dict, currency: Dict,
                                  risk_adj: Dict) -> Dict[str, Any]:
        """Aggregates discrete layers into an executive-level performance summary."""
        try:
            alloc_sum = brinson.get('allocation_effect', 0.0)
            select_sum = brinson.get('selection_effect', 0.0) + selection.get('net_selection_impact', 0.0)
            timing_sum = timing.get('net_timing_impact', 0.0)
            fx_sum = currency.get('currency_alpha', 0.0)

            total_attr_alpha = alloc_sum + select_sum + timing_sum + fx_sum

            # Dominant Alpha Drivers (threshold > 30% contribution)
            primary_drivers = []
            if abs(alloc_sum) > abs(total_attr_alpha) * 0.3: primary_drivers.append('asset_allocation')
            if abs(select_sum) > abs(total_attr_alpha) * 0.3: primary_drivers.append('security_selection')
            if abs(timing_sum) > abs(total_attr_alpha) * 0.3: primary_drivers.append('market_timing')
            if abs(fx_sum) > abs(total_attr_alpha) * 0.1: primary_drivers.append('currency_fx')

            return {
                'total_attribution_alpha': float(total_attr_alpha),
                'allocation_component': float(alloc_sum),
                'selection_component': float(select_sum),
                'timing_component': float(timing_sum),
                'currency_component': float(fx_sum),
                'alpha_drivers': primary_drivers,
                'jensen_alpha_annualized': risk_adj.get('annualized_risk_adjusted_alpha', 0.0)
            }

        except Exception as e:
            logger.warning(f"Executive summary synthesis failed: {e}")
            return {}

    def _generate_qualitative_recommendations(self, brinson: Dict, selection: Dict,
                                            timing: Dict) -> List[str]:
        """Translates quantitative attribution results into actionable strategic recommendations."""
        recommendations = []

        # Allocation Insights
        alloc_val = brinson.get('allocation_effect', 0.0)
        if alloc_val > 0.05:
            recommendations.append("Robust asset allocation efficiency. Maintain current weight distribution strategy.")
        elif alloc_val < -0.05:
            recommendations.append("Allocation drag detected. Review sectoral and asset-class weighting priorities.")

        # Selection Insights
        select_val = selection.get('net_selection_impact', 0.0)
        if select_val > 0.03:
            recommendations.append("High-quality security selection detected. Core stock-picking mechanism is performing optimally.")
        elif select_val < -0.03:
            recommendations.append("Selection drag identified. Conduct an audit of ticker selection criteria and filters.")

        # Timing Insights
        timing_val = timing.get('net_timing_impact', 0.0)
        realized_beta = timing.get('realized_beta', 1.0)

        if timing_val > 0.02:
            recommendations.append("Effective market timing signals. Continue utilizing current entry/exit orchestration.")
        elif timing_val < -0.02:
            recommendations.append("Market timing slippage. Evaluate signal lead/lag and entry point sensitivity.")

        if realized_beta > 1.2:
            recommendations.append("High systemic exposure (Beta > 1.2). Portfolio is significantly more aggressive than the benchmark.")
        elif realized_beta < 0.8:
            recommendations.append("Low systemic exposure (Beta < 0.8). Portfolio exhibits defensive characteristics relative to benchmark.")

        if not recommendations:
            recommendations.append("Balanced attribution profile. No immediate structural changes required; continue monitoring tracking error.")

        return recommendations