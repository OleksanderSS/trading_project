"""
Elite Risk Management System
- Historical Simulation VaR
- Parametric VaR (Normal + t-distribution)
- Monte Carlo VaR (bootstrap)
- GARCH dynamic volatility VaR
- Cornish-Fisher VaR (skewness + kurtosis)
- Ensemble VaR (combines multiple methods)
- Stress Testing (5 market scenarios)
- Liquidity Risk Assessment
- Risk Limits Management
"""
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class EliteRiskMetrics:
    """
    Elite Risk Management System - Unified risk calculation and monitoring

    Combines:
    - Multiple VaR methods (Historical, Parametric, Monte Carlo, GARCH, Cornish-Fisher)
    - Stress testing framework
    - Liquidity risk assessment
    - Risk limits enforcement
    """
    DEFAULT_VAR_LOSS = 0.05
    DEFAULT_CVAR_LOSS = 0.08

    def __init__(self, config_manager=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager
        self.returns_history = {}
        self.volatility_estimates = {}
        self._load_risk_limits()
        self._init_stress_scenarios()
        self.logger.info(
            'EliteRiskMetrics initialized with full risk management capabilities'
            )

    def _load_risk_limits(self):
        """Load risk limits from configuration"""
        if self.config_manager:
            risk_config = self.config_manager.get('strategy.risk_management',
                {})
            self.limits = {'max_portfolio_var': risk_config.get(
                'max_portfolio_var_pct', 0.05), 'max_single_position':
                risk_config.get('max_single_position_pct', 0.1),
                'max_daily_loss': risk_config.get('max_daily_loss_pct',
                0.03), 'max_drawdown': risk_config.get('max_drawdown_pct',
                0.15), 'max_leverage': risk_config.get('max_leverage', 2.0)}
        else:
            self.limits = {'max_portfolio_var': 0.05, 'max_single_position':
                0.1, 'max_daily_loss': 0.03, 'max_drawdown': 0.15,
                'max_leverage': 2.0}

    def _init_stress_scenarios(self):
        """Initialize stress testing scenarios"""
        self.stress_scenarios = {'market_crash': {'shock': -0.15,
            'description': 'Sudden 15% broad market crash'},
            'volatility_spike': {'volatility_multiplier': 3.0,
            'description': 'Extreme 3x volatility expansion'},
            'liquidity_crisis': {'liquidity_dryup': 0.8, 'description':
            'Severe 80% market liquidity reduction'}, 'interest_rate_shock':
            {'rate_change': 0.025, 'description':
            'Unexpected 2.5% interest rate hike'}, 'correlated_crash': {
            'correlation_increase': 0.8, 'description':
            'Extreme asset correlation breakdown'}}

    def _clean_recent_returns(self, ticker: str, min_samples: int=30,
        lookback_days: int | None=None) ->pd.Series | None:
        if ticker not in self.returns_history:
            return None
        returns = pd.Series(self.returns_history[ticker], dtype=float).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if lookback_days is not None and len(returns) > lookback_days:
            returns = returns.iloc[-lookback_days:]
        if len(returns) < min_samples:
            return None
        return returns

    def _fallback_var_result(self, method: str, confidence_level: float,
        time_horizon: int=1) ->dict[str, float | str]:
        return {'var': self.DEFAULT_VAR_LOSS, 'cvar': self.DEFAULT_CVAR_LOSS,
            'confidence': confidence_level, 'time_horizon': time_horizon,
            'method': method, 'status': 'insufficient_data'}

    def compute_historical_simulation_var(self, ticker: str,
        confidence_level: float=0.95, lookback_days: int=252) ->float:
        """
        Historical Simulation VaR

        The simplest (but effective) - take historical returns,
        sort, take the percentile

        Args:
            confidence_level: 0.95 = 95% VaR = 5% tail loss
            lookback_days: How far back to look

        Returns:
            VaR (as percentage, e.g., 0.03 = 3% loss)
        """
        recent_returns = self._clean_recent_returns(ticker, min_samples=30,
            lookback_days=lookback_days)
        if recent_returns is None:
            return self.DEFAULT_VAR_LOSS
        tail_percentile = (1 - confidence_level) * 100
        var_loss_positive = max(0.0, float(-np.percentile(  # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
            recent_returns, tail_percentile)))
        return float(max(0.001, var_loss_positive))

    def compute_cornish_fisher_var(self, ticker: str, confidence_level:
        float=0.95, lookback_days: int=252) ->tuple[float, float]:
        """
        Cornish-Fisher VaR (accounts for skewness and kurtosis)

        More accurate than normal VaR, especially for tails

        Returns:
            (VaR, CVaR)  - Value at Risk & Conditional VaR (Expected Shortfall)
        """
        recent_returns = self._clean_recent_returns(ticker, min_samples=30,
            lookback_days=lookback_days)
        if recent_returns is None:
            return self.DEFAULT_VAR_LOSS, self.DEFAULT_CVAR_LOSS
        mean = recent_returns.mean()
        std = recent_returns.std()
        skewness = stats.skew(recent_returns)
        kurtosis = stats.kurtosis(recent_returns)
        z = stats.norm.ppf(1 - confidence_level)
        z_cf = z + (z ** 2 - 1) * skewness / 6 + (z ** 3 - 3 * z
            ) * kurtosis / 24 - (2 * z ** 3 - 5 * z) * skewness ** 2 / 36
        var_cf = -(mean + z_cf * std)
        cvar_cf = var_cf * (1 + abs(kurtosis) / 4)
        return max(0.001, var_cf), max(0.001, cvar_cf)

    def compute_garch_var(self, ticker: str, confidence_level: float=0.95
        ) ->float:
        """
        GARCH(1,1) dynamic volatility VaR

        GARCH captures volatility clustering
        σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

        Returns:
            Predicted VaR for tomorrow # audit-ignore: no leakage, risk estimation
        """
        returns = self._clean_recent_returns(ticker, min_samples=50,
            lookback_days=252)
        if returns is None:
            return self.DEFAULT_VAR_LOSS
        try:
            from arch import arch_model
        except ImportError:
            self.logger.warning(
                'ARCH package not installed. Falling back to simple VaR.')
            return self.compute_historical_simulation_var(ticker,
                confidence_level)
        try:
            model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            tomorrow_vol = float(forecast.variance.values[-1, 0] ** 0.5) / 100 # audit-ignore: no leakage, risk estimation
            z = stats.norm.ppf(1 - confidence_level)
            var_value: float = float(-tomorrow_vol * z) # audit-ignore: no leakage, risk estimation
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'GARCH VaR for {ticker}: {var_value:.3%}')
            return float(max(0.001, var_value))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'GARCH failed for {ticker}: {e}. Using historical VaR.')
            return self.compute_historical_simulation_var(ticker,
                confidence_level)

    def compute_comprehensive_risk_metrics(self, ticker: str, position_size:
        int, entry_price: float, portfolio_value: float) ->dict:
        """
        Compute comprehensive set of risk metrics

        Returns:
            Dict with VaR, CVaR, portfolio impact, etc.
        """
        position_value = position_size * entry_price
        position_pct = (position_value / portfolio_value if portfolio_value >
            0 else 0)
        hs_var = self.compute_historical_simulation_var(ticker, 0.95)
        cf_var, cf_cvar = self.compute_cornish_fisher_var(ticker, 0.95)
        garch_var = self.compute_garch_var(ticker, 0.95)
        ensemble_var = np.mean([hs_var, cf_var, garch_var])
        var_dollars = position_value * ensemble_var
        cvar_dollars = position_value * cf_cvar
        portfolio_var = portfolio_value * ensemble_var
        return {'ticker': ticker, 'position_size': position_size,
            'position_value': position_value, 'position_pct': position_pct,
            'var_95_pct': {'historical_simulation': hs_var,
            'cornish_fisher': cf_var, 'garch': garch_var, 'ensemble':
            ensemble_var}, 'cvar_95_pct': cf_cvar, 'var_95_dollars':
            var_dollars, 'cvar_95_dollars': cvar_dollars,
            'portfolio_var_impact': portfolio_var, 'status': 'ok' if
            var_dollars < portfolio_value * 0.05 else 'high_risk'}

    def update_returns(self, ticker: str, returns: pd.Series):
        """Update historical data"""
        self.returns_history[ticker] = returns

    def get_risk_report(self, positions: dict[str, int], prices: dict[str,
        float], portfolio_value: float) ->dict:
        """
        Generate risk report for the portfolio
        """
        total_position_var = 0.0
        total_position_cvar = 0.0
        position_risks = []
        for ticker, qty in positions.items():
            if qty == 0 or ticker not in prices:
                continue
            risk_metrics = self.compute_comprehensive_risk_metrics(ticker,
                qty, prices[ticker], portfolio_value)
            total_position_var += risk_metrics['var_95_dollars']
            total_position_cvar += risk_metrics['cvar_95_dollars']
            position_risks.append(risk_metrics)
        portfolio_var_pct = (total_position_var / portfolio_value if
            portfolio_value > 0 else 0)
        portfolio_cvar_pct = (total_position_cvar / portfolio_value if
            portfolio_value > 0 else 0)
        return {'portfolio_var_95_dollars': total_position_var,
            'portfolio_var_95_pct': portfolio_var_pct,
            'portfolio_cvar_95_dollars': total_position_cvar,
            'portfolio_cvar_95_pct': portfolio_cvar_pct, 'position_risks':
            position_risks, 'timestamp': datetime.now().isoformat(),
            'risk_status': self._determine_risk_status(portfolio_var_pct)}

    def _determine_risk_status(self, portfolio_var_pct: float) ->str:
        """Determine risk status based on portfolio variance percentage."""
        if portfolio_var_pct < 0.05:
            return 'ok'
        if portfolio_var_pct < 0.1:
            return 'elevated'
        return 'high'

    def compute_parametric_var(self, ticker: str, confidence_level: float=
        0.95, time_horizon: int=1, distribution: str='normal') ->dict[str,
        float]:
        """
        Parametric VaR using Normal or Student's t distribution

        Args:
            ticker: Asset ticker
            confidence_level: Confidence level (0.95, 0.99)
            time_horizon: Time horizon in days
            distribution: 'normal' or 't' distribution

        Returns:
            Dict with VaR, CVaR, and parameters (values can be float or str)
        """
        returns = self._clean_recent_returns(ticker, min_samples=30)
        if returns is None:
            return self._fallback_var_result('parametric_fallback',
                confidence_level, time_horizon)
        mu = returns.mean()
        sigma = returns.std()
        if distribution == 'normal':
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mu * time_horizon + sigma * np.sqrt(time_horizon) * z_score
            alpha = 1 - confidence_level
            cvar = mu * time_horizon - sigma * np.sqrt(time_horizon
                ) * stats.norm.pdf(z_score) / alpha
        elif distribution == 't':
            from scipy.stats import t as t_dist
            df_param, loc, scale = t_dist.fit(returns)
            t_score = t_dist.ppf(1 - confidence_level, df_param)
            var = loc * time_horizon + scale * np.sqrt(time_horizon) * t_score
            alpha = 1 - confidence_level
            density = t_dist.pdf(t_score, df_param)
            if df_param > 1:
                es_factor = (df_param + t_score**2) / (df_param - 1) * (density / alpha)
                cvar = loc * time_horizon - scale * np.sqrt(time_horizon) * es_factor
            else:
                cvar = var
        else:
            raise ValueError(f'Unsupported distribution: {distribution}')
        var_loss_positive = max(0.0, float(-var))
        cvar_loss_positive = max(0.0, float(-cvar))
        return {'var': var_loss_positive, 'cvar': cvar_loss_positive,
            'var_return_threshold': float(var), 'cvar_return_threshold':
            float(cvar), 'confidence': confidence_level, 'time_horizon':
            time_horizon, 'method': f'parametric_{distribution}', 'mu':
            float(mu), 'sigma': float(sigma), 'status': 'ok'}

    def compute_monte_carlo_var(self, ticker: str, confidence_level: float=
        0.95, time_horizon: int=1, n_simulations: int=10000) ->dict[str, float
        ]:
        """
        Monte Carlo VaR simulation via bootstrap sampling

        Args:
            ticker: Asset ticker
            confidence_level: Confidence level
            time_horizon: Forecast horizon in days
            n_simulations: Number of simulations

        Returns:
            Dict with VaR, CVaR, and simulation stats
        """
        returns = self._clean_recent_returns(ticker, min_samples=30)
        if returns is None:
            return self._fallback_var_result('monte_carlo_fallback',
                confidence_level, time_horizon)
        rng = np.random.default_rng(42)
        simulated_returns = []
        for _ in range(n_simulations):
            sample = rng.choice(returns.values, size=time_horizon, replace=True
                )
            portfolio_return = np.prod(1 + sample) - 1
            simulated_returns.append(portfolio_return)
        simulated_returns = np.array(simulated_returns)
        var_loss_positive = max(0.0, float(-np.percentile(  # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
            simulated_returns, (1 - confidence_level) * 100)))
        tail_returns = simulated_returns[simulated_returns <=
            -var_loss_positive]
        cvar_loss_positive = (max(0.0, float(-tail_returns.mean())) if len(
            tail_returns) > 0 else var_loss_positive)
        return {'var': var_loss_positive, 'cvar': cvar_loss_positive,
            'confidence': confidence_level, 'time_horizon': time_horizon,
            'method': 'monte_carlo', 'status': 'ok', 'n_simulations':
            n_simulations, 'mean_simulated': float(np.mean(simulated_returns
            )), 'std_simulated': float(np.std(simulated_returns))}

    def run_stress_test(self, portfolio: dict[str, float], scenario: str=
        'market_crash') ->dict[str, Any]:
        """
        Run stress test simulation for portfolio

        Args:
            portfolio: Dict of {ticker: weight}
            scenario: Scenario name ('market_crash', 'volatility_spike', etc.)

        Returns:
            Impact analysis and recommendations
        """
        if scenario not in self.stress_scenarios:
            raise ValueError(f'Unknown stress scenario: {scenario}')
        scenario_config = self.stress_scenarios[scenario]
        if scenario == 'market_crash':
            shock = float(scenario_config['shock'])
            portfolio_impact = sum(float(weight) * shock for weight in
                portfolio.values())
        elif scenario == 'volatility_spike':
            vol_multiplier = float(scenario_config['volatility_multiplier'])
            base_var = 0.02
            stressed_var = base_var * vol_multiplier
            portfolio_impact = -stressed_var * 2
        elif scenario == 'liquidity_crisis':
            liquidity_reduction = float(scenario_config['liquidity_dryup'])
            portfolio_impact = -0.05 * liquidity_reduction
        else:
            portfolio_impact = -0.05
        recovery_days: float = abs(portfolio_impact) * 100
        return {'scenario': scenario, 'description': scenario_config[
            'description'], 'portfolio_impact': float(portfolio_impact),
            'portfolio_loss_pct': float(abs(portfolio_impact) * 100),
            'estimated_recovery_days': int(recovery_days),
            'breaches_limits': abs(portfolio_impact) > 0.1,
            'recommendations': self._generate_stress_recommendations(
            portfolio_impact, scenario)}

    def _generate_stress_recommendations(self, impact: float, scenario: str
        ) ->list[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        if abs(impact) > 0.1:
            recommendations.append(
                'Critical: Consider partial position liquidation')
            recommendations.append('Recommended: Tighten stop-loss levels')
        if scenario == 'market_crash':
            recommendations.append(
                'Action: Consider hedging via options or inverse ETFs')
            recommendations.append('Action: Re-evaluate momentum-heavy assets')
        elif scenario == 'volatility_spike':
            recommendations.append(
                'Action: Reduce position sizes to lower volatility')
            recommendations.append(
                'Action: Implement volatility-targeting strategies')
        return recommendations

    def assess_liquidity_risk(self, ticker: str, volume_data: pd.Series,
        price_data: pd.Series, position_size: float) ->dict[str, Any]:
        """
        Assess liquidity risk for an asset

        Args:
            ticker: Asset ticker
            volume_data: Historical trading volumes
            price_data: Historical prices
            position_size: Planned position size in USD

        Returns:
            Liquidity metrics and recommendations
        """
        avg_daily_volume = volume_data.mean()
        avg_daily_volume_dollars = (volume_data * price_data).mean()
        returns = price_data.pct_change(fill_method=None)
        volatility = returns.std()
        spread_estimate = volatility * 0.01
        market_impact_pct = min(position_size / avg_daily_volume_dollars, 0.1)
        volume_score = min(avg_daily_volume / 1000000, 1.0)
        spread_score = max(0, 1 - spread_estimate * 100)
        liquidity_score = (volume_score * 0.7 + spread_score * 0.3) * 100
        if liquidity_score < 30:
            risk_level = 'HIGH'
            risk_description = 'High liquidity risk - avoid large positions'
        elif liquidity_score < 60:
            risk_level = 'MEDIUM'
            risk_description = 'Moderate liquidity risk - limit position sizes'
        else:
            risk_level = 'LOW'
            risk_description = (
                'Low liquidity risk - suitable for standard trading')
        return {'ticker': ticker, 'liquidity_score': float(liquidity_score),
            'risk_level': risk_level, 'risk_description': risk_description,
            'avg_daily_volume': float(avg_daily_volume),
            'avg_daily_volume_dollars': float(avg_daily_volume_dollars),
            'estimated_spread_pct': float(spread_estimate),
            'market_impact_pct': float(market_impact_pct),
            'max_position_size': float(avg_daily_volume_dollars * 0.01),
            'recommendations': self._generate_liquidity_recommendations(
            risk_level, position_size, avg_daily_volume_dollars)}

    def _generate_liquidity_recommendations(self, risk_level: str,
        position_size: float, avg_daily_volume: float) ->list[str]:
        """Generate liquidity recommendations"""
        recommendations = []
        if risk_level == 'HIGH':
            recommendations.append(
                'Avoid this asset or use extremely small positions')
            recommendations.append('Consider more liquid alternatives')
        elif risk_level == 'MEDIUM':
            max_safe_size = avg_daily_volume * 0.005
            if position_size > max_safe_size:
                recommendations.append(
                    f'Reduce position below ${max_safe_size:,.0f}')
        return recommendations

    def check_limits(self, portfolio_value: float, positions: dict[str,
        dict[str, Any]], daily_pnl: float, current_drawdown: float) ->dict[
        str, Any]:
        """
        Check if portfolio adheres to risk limits.

        Previously used ``estimated_var = portfolio_value * 0.02`` (a hard-coded
        2% flat estimate) which always evaluated to less than the default
        ``max_portfolio_var`` limit of 5%, so the portfolio-VaR check never
        triggered.  Now delegates to ``compute_comprehensive_risk_metrics`` for
        each position and aggregates to a real ensemble-VaR figure — consistent
        with ``get_risk_report``.
        """
        violations = []
        warnings = []

        # --- Portfolio VaR (ensemble of HS / Cornish-Fisher / GARCH) ---
        total_var_dollars = 0.0
        for ticker, pos_data in positions.items():
            pos_value = pos_data.get('value', 0.0)
            if pos_value <= 0 or ticker not in self.returns_history:
                # Fallback: use DEFAULT_VAR_LOSS when no history available
                total_var_dollars += pos_value * self.DEFAULT_VAR_LOSS
                continue
            try:
                entry_price = pos_data.get('entry_price', pos_data.get('price', 1.0))
                position_size = int(pos_value / entry_price) if entry_price > 0 else 0
                if position_size > 0:
                    risk = self.compute_comprehensive_risk_metrics(
                        ticker, position_size, entry_price, portfolio_value
                    )
                    total_var_dollars += risk['var_95_dollars']
                else:
                    total_var_dollars += pos_value * self.DEFAULT_VAR_LOSS
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"check_limits: could not compute VaR for {ticker}: {e}")
                total_var_dollars += pos_value * self.DEFAULT_VAR_LOSS

        estimated_var_pct = total_var_dollars / portfolio_value if portfolio_value > 0 else 0.0

        if estimated_var_pct > self.limits['max_portfolio_var']:
            violations.append({
                'type': 'portfolio_var',
                'current': estimated_var_pct,
                'limit': self.limits['max_portfolio_var'],
                'message': f'Portfolio VaR {estimated_var_pct:.1%} exceeds limit {self.limits["max_portfolio_var"]:.1%}'
            })

        # --- Concentration limits ---
        for ticker, pos_data in positions.items():
            concentration = pos_data['value'] / portfolio_value
            if concentration > self.limits['max_single_position']:
                violations.append({'type': 'single_position', 'ticker':
                    ticker, 'current': concentration, 'limit': self.limits[
                    'max_single_position'], 'message':
                    f'Position {ticker} concentration {concentration:.1%} exceeds limit'
                    })

        # --- Daily loss limit ---
        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        if daily_loss_pct > self.limits['max_daily_loss']:
            violations.append({'type': 'daily_loss', 'current':
                daily_loss_pct, 'limit': self.limits['max_daily_loss'],
                'message': f'Daily loss {daily_loss_pct:.1%} exceeds limit'})

        # --- Drawdown limit ---
        if current_drawdown > self.limits['max_drawdown']:
            violations.append({'type': 'drawdown', 'current':
                current_drawdown, 'limit': self.limits['max_drawdown'],
                'message': f'Drawdown {current_drawdown:.1%} exceeds limit'})

        # --- Warnings (approaching thresholds) ---
        if estimated_var_pct > self.limits['max_portfolio_var'] * 0.8:
            warnings.append('Portfolio VaR approaching critical threshold')
        if daily_loss_pct > self.limits['max_daily_loss'] * 0.7:
            warnings.append('Daily P&L approaching stop-loss threshold')

        return {'limits_respected': len(violations) == 0, 'violations':
            violations, 'warnings': warnings, 'checked_at': datetime.now().
            isoformat(), 'portfolio_value': portfolio_value,
            'estimated_var_pct': estimated_var_pct,
            'positions_count': len(positions)}
