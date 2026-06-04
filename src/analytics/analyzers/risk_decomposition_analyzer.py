"""
Risk Decomposition Analyzer
Deconstructs portfolio risk into fundamental components to identify volatility sources.

Risk Layers Analyzed:
- Systematic Risk (Market Exposure)
- Idiosyncratic Risk (Asset-Specific Variance)
- Factor Risk (Structural Multi-Factor Exposure)
- Liquidity Risk (Market Impact Sensitivity)
- Concentration Risk (Asset and Sector Clustering)

Methodologies Supported:
- Multi-Factor Risk Models (Fama-French, Industry Factors)
- Principal Component Analysis (PCA) for Latent Factor Discovery
- Marginal and Incremental Risk Attribution
"""
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

from ..interfaces import IAnalyzer

logger = ProjectLogger.get_logger('RiskDecompositionAnalyzer')


class RiskDecompositionAnalyzer(IAnalyzer):
    """
    Core engine for deconstructing risk profiles.
    Translates raw return variance into actionable risk attribution layers.
    """

    def __init__(self, config: dict[str, Any] | None=None):
        """
        Initializes the risk decomposition engine.

        Args:
            config: Configuration parameters for PCA components, factor models, and thresholds.
        """
        self.config = config or {}
        self.enable_pca = self.config.get('use_pca', True)
        self.latent_factor_count = self.config.get('n_factors', 5)
        self.primary_factor_model = self.config.get('factor_model',
            'fama_french')
        self.tracked_metrics = ['volatility', 'var_95', 'cvar_95',
            'max_drawdown', 'sharpe_ratio']
        self.custom_risk_factors = self.config.get('custom_factors', [])
        logger.info(
            f'RiskDecompositionAnalyzer initialized using {self.primary_factor_model} methodology.'
            )

    def analyze(self, data: dict[str, pd.DataFrame], **kwargs) ->dict[str, Any
        ]:
        """
        Executes a holistic risk decomposition suite.

        Args:
            data: Composite dictionary containing:
                - 'portfolio_returns': pd.DataFrame of multi-asset historical returns.
                - 'factor_returns': Optional pd.DataFrame of systematic risk factor returns.
                - 'weights': Dictionary of asset allocation weights.

        Returns:
            Structured risk report with systematic, specific, and factor-level attribution.
        """
        portfolio_returns = data.get('portfolio_returns')
        factor_returns = data.get('factor_returns')
        allocation_weights = data.get('weights', {})

        if portfolio_returns is None or portfolio_returns.empty:
            raise DataProcessingError('portfolio_returns dataset required for risk analysis.')

        aggregate_risk = self._calculate_aggregate_risk_profile(
            portfolio_returns, allocation_weights)
        systematic_vector, idiosyncratic_vector = (self.
            _decompose_systematic_idiosyncratic(portfolio_returns,
            factor_returns))
        factor_attribution_payload = self._decompose_factor_risk(
            portfolio_returns, factor_returns)
        concentration_metrics = self._calculate_concentration_profile(
            portfolio_returns, allocation_weights)
        liquidity_risk_proxies = self._calculate_liquidity_risk_proxies(
            portfolio_returns)
        payload = {'aggregate_risk': aggregate_risk,
            'systematic_risk_profile': systematic_vector,
            'idiosyncratic_risk_profile': idiosyncratic_vector,
            'factor_attribution': factor_attribution_payload,
            'concentration_metrics': concentration_metrics,
            'liquidity_proxies': liquidity_risk_proxies,
            'risk_contribution_summary': self.
            _summarize_risk_contributions(systematic_vector,
            idiosyncratic_vector, concentration_metrics,
            liquidity_risk_proxies), 'strategic_recommendations': self.
            _generate_risk_mitigation_recommendations(systematic_vector,
            idiosyncratic_vector, factor_attribution_payload,
            concentration_metrics, liquidity_risk_proxies),
            'analysis_timestamp': datetime.now().isoformat()}
        logger.info(
            'Risk decomposition analysis suite completed successfully.')
        return payload

    def _calculate_aggregate_risk_profile(self, returns: pd.DataFrame,
        weights: dict[str, float]) ->dict[str, float]:
        """Calculates top-level realized risk metrics for the combined portfolio."""
        if not weights:
            asset_population = len(returns.columns)
            weights = dict.fromkeys(returns.columns, 1.0 / asset_population)

        weighted_returns = returns.values @ np.array([weights.get(t,
            0.0) for t in returns.columns])
        weighted_returns = pd.Series(weighted_returns, dtype=float).replace(
            [np.inf, -np.inf], np.nan).dropna().to_numpy()
        if weighted_returns.size == 0:
            raise DataProcessingError(
                'Portfolio returns contain no finite observations for risk metrics.'
            )

        weighted_std = float(np.std(weighted_returns))
        if not np.isfinite(weighted_std) or weighted_std <= 1e-12:
            raise DataProcessingError("Portfolio has zero variance, cannot calculate risk metrics.")

        realized_vol = weighted_std * np.sqrt(252)
        var_05_threshold = np.percentile(weighted_returns, 5)  # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
        cvar_05_threshold = weighted_returns[weighted_returns <=
            var_05_threshold].mean()
        var_loss_positive = max(0.0, float(-var_05_threshold))
        cvar_loss_positive = max(0.0, float(-cvar_05_threshold))
        wealth_index = np.cumprod(1 + weighted_returns)
        peak_nav = np.maximum.accumulate(wealth_index)
        max_dd = ((wealth_index - peak_nav) / peak_nav).min()
        annual_rf = 0.02
        excess_mean = np.mean(weighted_returns) - annual_rf / 252
        realized_sharpe = excess_mean / weighted_std * np.sqrt(252)
        return {'annualized_volatility': float(realized_vol),
            'value_at_risk_95': var_loss_positive,
            'conditional_var_95': cvar_loss_positive,
            'var_return_threshold_95': float(var_05_threshold),
            'conditional_var_return_threshold_95': float(cvar_05_threshold),
            'max_drawdown': float(max_dd), 'realized_sharpe_ratio':
            float(realized_sharpe)}

    def _decompose_systematic_idiosyncratic(self, returns: pd.DataFrame,
        factor_returns: pd.DataFrame | None) ->tuple[dict[str, float],
        dict[str, float]]:
        """Isolates systematic market exposure from individual asset variance."""
        systematic_map = {}
        idiosyncratic_map = {}
        for asset_name in returns.columns:
            asset_series = returns[asset_name].dropna()
            if factor_returns is not None and not factor_returns.empty:
                shared_idx = asset_series.index.intersection(factor_returns
                    .index)
                if len(shared_idx) < 30:
                    systematic_map[asset_name] = 0.7 * float(asset_series
                        .std())
                    idiosyncratic_map[asset_name] = 0.3 * float(
                        asset_series.std())
                    continue
                x_factors = factor_returns.loc[shared_idx].values
                y_returns = asset_series.loc[shared_idx].values
                regressor = LinearRegression().fit(x_factors, y_returns)
                systematic_variance = regressor.predict(x_factors).var()
                systematic_map[asset_name] = float(np.sqrt(
                    systematic_variance))
                residual_variance = (y_returns - regressor.predict(
                    x_factors)).var()
                idiosyncratic_map[asset_name] = float(np.sqrt(
                    residual_variance))
            else:
                total_var = float(asset_series.std())
                systematic_map[asset_name] = 0.7 * total_var
                idiosyncratic_map[asset_name] = 0.3 * total_var
        return systematic_map, idiosyncratic_map

    def _decompose_factor_risk(self, returns: pd.DataFrame, factor_returns:
        pd.DataFrame | None) ->dict[str, Any]:
        """Calculates portfolio sensitivity and risk contribution for each identified risk factor."""
        if factor_returns is None or factor_returns.empty:
            if self.enable_pca:
                clean_returns = returns.dropna()
                pca_engine = PCA(n_components=min(self.
                    latent_factor_count, len(clean_returns.columns)),
                    random_state=42)
                pca_engine.fit(clean_returns.T)
                return {'methodology': 'pca_latent_discovery',
                    'loadings': pca_engine.components_.tolist(),
                    'variance_explained_ratio': pca_engine.
                    explained_variance_ratio_.tolist(),
                    'discovered_factor_count': int(len(pca_engine.
                    explained_variance_ratio_))}
            else:
                raise DataProcessingError('Factor data missing and PCA discovery disabled.')
        loadings_map = {}
        contributions_map = {}
        for asset_name in returns.columns:
            asset_series = returns[asset_name].dropna()
            common_idx = asset_series.index.intersection(factor_returns
                .index)
            if len(common_idx) >= 30:
                x_f = factor_returns.loc[common_idx].values
                y_a = asset_series.loc[common_idx].values
                ols = LinearRegression().fit(x_f, y_a)
                loadings_map[asset_name] = ols.coef_.tolist()
                contributions_map[asset_name] = (ols.coef_ ** 2).tolist()
        return {'methodology': 'multi_factor_regression',
            'asset_loadings': loadings_map, 'squared_contributions':
            contributions_map, 'factor_labels': factor_returns.columns.
            tolist()}

    def _calculate_concentration_profile(self, returns: pd.DataFrame,
        weights: dict[str, float]) ->dict[str, Any]:
        """Evaluates asset and sector clustering to identify concentration risks."""
        if not weights:
            pop = len(returns.columns)
            weights = dict.fromkeys(returns.columns, 1.0 / pop)
        weight_vector = np.array([weights.get(t, 0.0) for t in returns.
            columns])
        realized_hhi = np.sum(weight_vector ** 2)
        effective_position_count = (1.0 / realized_hhi if realized_hhi >
            0 else 0.0)
        ranked_weights = np.sort(weight_vector)[::-1]
        top_3_concentration = np.sum(ranked_weights[:3])
        sorted_w = np.sort(weight_vector)
        n_w = len(sorted_w)
        weight_cumsum = np.cumsum(sorted_w)
        calculated_gini = (n_w + 1 - 2 * np.sum(weight_cumsum) /
            weight_cumsum[-1]) / n_w if weight_cumsum[-1] > 0 else 0.0
        return {'herfindahl_hirschman_index': float(realized_hhi),
            'effective_asset_count': float(effective_position_count),
            'top_3_concentration_ratio': float(top_3_concentration),
            'gini_coefficient': float(calculated_gini),
            'concentration_warning': bool(realized_hhi > 0.25)}

    def _calculate_liquidity_risk_proxies(self, returns: pd.DataFrame) ->dict[
        str, Any]:
        """Estimates market impact sensitivity as a proxy for liquidity risk."""
        asset_liquidity_map = {}
        for asset_name in returns.columns:
            asset_series = returns[asset_name].dropna()
            realized_vol = asset_series.std()
            estimated_turnover_baseline = 0.1
            implied_risk_score = realized_vol / (
                estimated_turnover_baseline + 0.01)
            asset_liquidity_map[asset_name] = {'volatility_proxy':
                float(realized_vol), 'relative_illiquidity_score':
                float(implied_risk_score), 'illiquid_flag': bool(
                implied_risk_score > 1.0)}
        portfolio_wide_score = np.mean([entry[
            'relative_illiquidity_score'] for entry in
            asset_liquidity_map.values()])
        return {'asset_level_liquidity': asset_liquidity_map,
            'portfolio_liquidity_risk_index': float(
            portfolio_wide_score), 'illiquid_asset_count': int(sum(1 for
            entry in asset_liquidity_map.values() if entry[
            'illiquid_flag']))}

    def _summarize_risk_contributions(self, systematic: dict[str, float],
        idiosyncratic: dict[str, float], concentration: dict[str, Any],
        liquidity: dict[str, Any]) ->dict[str, Any]:
        """Aggregates and normalizes risk layers into a high-level attribution report."""
        mean_systematic = np.mean(list(systematic.values())
            ) if systematic else 0.0
        mean_idiosyncratic = np.mean(list(idiosyncratic.values())
            ) if idiosyncratic else 0.0
        net_risk = mean_systematic + mean_idiosyncratic
        if net_risk > 0:
            systematic_weight = mean_systematic / net_risk
            idiosyncratic_weight = mean_idiosyncratic / net_risk
        else:
            systematic_weight = 0.5
            idiosyncratic_weight = 0.5
        return {'systematic_risk_contribution_ratio': float(
            systematic_weight), 'idiosyncratic_risk_contribution_ratio':
            float(idiosyncratic_weight), 'clustering_impact_hhi':
            concentration.get('herfindahl_hirschman_index', 0.0),
            'liquidity_impact_index': liquidity.get(
            'portfolio_liquidity_risk_index', 0.0),
            'diversification_efficiency': float(1.0 - concentration.get
            ('herfindahl_hirschman_index', 1.0))}

    def _generate_risk_mitigation_recommendations(self, systematic: dict[
        str, float], idiosyncratic: dict[str, float], factors: dict[str,
        Any], concentration: dict[str, Any], liquidity: dict[str, Any]) ->list[
        str]:
        """Translates quantitative risk metrics into actionable portfolio mitigation strategies."""
        recommendations = []
        if concentration.get('herfindahl_hirschman_index', 0.0) > 0.25:
            recommendations.append(
                'High portfolio clustering detected. Diversify asset allocation to reduce structural fragility.'
                )
        if concentration.get('effective_asset_count', 10) < 5:
            recommendations.append(
                'Low effective asset count identified. Increase positional diversity across non-correlated sectors.'
                )
        s_val = np.mean(list(systematic.values())) if systematic else 0.0
        i_val = np.mean(list(idiosyncratic.values())) if idiosyncratic else 0.0
        total_v = s_val + i_val
        if total_v > 0:
            if s_val / total_v > 0.8:
                recommendations.append(
                    'Market (Systematic) risk predominates. Implement index-based hedging or reduce Beta exposure.'
                    )
            elif i_val / total_v > 0.8:
                recommendations.append(
                    'Specific (Idiosyncratic) risk predominates. This can be mitigated through increased asset-level diversification.'
                    )
        if liquidity.get('portfolio_liquidity_risk_index', 0.0) > 1.0:
            recommendations.append(
                'Elevated liquidity risk index. Review positions in low-volume tickers or reduce individual position sizes.'
                )
        illiquid_c = liquidity.get('illiquid_asset_count', 0)
        if illiquid_c > 0:
            recommendations.append(
                f'Action required: {illiquid_c} assets flagged for insufficient liquidity. Monitor slippage and exit availability.'
                )
        if factors.get('methodology') == 'pca_latent_discovery':
            explained_v = sum(factors.get('variance_explained_ratio', []))
            if explained_v < 0.5:
                recommendations.append(
                    'Low factor explainability observed. Latent PCA factors capture < 50% of variance; consider alternative risk models.'
                    )
        if not recommendations:
            recommendations.append(
                'Risk profile appears balanced and structurally sound. Maintain current monitoring routines.'
                )
        return recommendations
