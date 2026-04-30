# src/risk_management/framework.py
"""
Risk Management Framework - Core system for risk modeling and limits enforcement.

Features:
- Value at Risk (VaR) calculations (Historical, Parametric, Monte Carlo)
- Conditional VaR (CVaR) / Expected Shortfall
- Stress Testing Framework (Scenario analysis)
- Liquidity Risk Assessment
- Risk Limits Management and Enforcement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.stats import norm, t
from datetime import datetime, timedelta
import logging

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager, get_current_config

logger = ProjectLogger.get_logger("RiskManagement")

class RiskManagementError(Exception):
    """Custom exception for risk management calculation errors."""
    pass

class VaRCalculator:
    """
    Value at Risk Calculator implementing multiple statistical methods.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """Initializes the calculator with global risk configurations."""
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("VaRCalculator")

        # Configuration parameters
        risk_config = self.config.get('strategy.risk_management', {})
        self.confidence_levels = risk_config.get('var_confidence_levels', [0.95, 0.99])
        self.time_horizon = risk_config.get('var_time_horizon_days', 1)
        self.annual_trading_days = risk_config.get('annual_trading_days', 252)
        self.random_seed = self.config.get('performance.random_seed', 42)

    def calculate_var_historical(self,
                                returns: pd.Series,
                                confidence: float = 0.95,
                                time_horizon: int = 1) -> Dict[str, float]:
        """
        Historical Simulation VaR.
        Determines risk by calculating percentiles from actual historical distribution.

        Args:
            returns: Historical return series.
            confidence: Confidence level (e.g., 0.95, 0.99).
            time_horizon: Time horizon in days for scaling.

        Returns:
            Dictionary containing VaR, CVaR, and audit parameters.
        """
        try:
            if len(returns) < 30:
                raise RiskManagementError("Insufficient data for reliable VaR calculation")

            # Percentile-based VaR calculation
            var_pct = np.percentile(returns, (1 - confidence) * 100)

            # Scale to time horizon
            if time_horizon > 1:
                # Assumes square-root-of-time scaling
                scaled_var = var_pct * np.sqrt(time_horizon)
            else:
                scaled_var = var_pct

            # Expected Shortfall (CVaR) - average of returns in the tail
            tail_returns = returns[returns <= var_pct]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var_pct

            return {
                'var': float(scaled_var),
                'cvar': float(cvar),
                'confidence': confidence,
                'time_horizon': time_horizon,
                'method': 'historical',
                'sample_size': len(returns),
                'tail_ratio': len(tail_returns) / len(returns)
            }

        except Exception as e:
            self.logger.error(f"Historical VaR calculation failure: {e}")
            return {'error': str(e)}

    def calculate_var_parametric(self,
                                returns: pd.Series,
                                confidence: float = 0.95,
                                time_horizon: int = 1,
                                distribution: str = 'normal') -> Dict[str, float]:
        """
        Parametric VaR using Normal or Student's t distribution.

        Args:
            returns: Historical return series.
            confidence: Confidence level.
            time_horizon: Time horizon in days.
            distribution: 'normal' or 't' distribution model.
        """
        try:
            if len(returns) < 30:
                raise RiskManagementError("Insufficient data for parametric fitting")

            mu = returns.mean()
            sigma = returns.std()

            if distribution == 'normal':
                # Normal distribution VaR (mean + sigma * z_score)
                z_score = norm.ppf(1 - confidence)
                var = mu * time_horizon + sigma * np.sqrt(time_horizon) * z_score

                # Analytical CVaR for normal distribution
                alpha = 1 - confidence
                cvar = mu * time_horizon - sigma * np.sqrt(time_horizon) * norm.pdf(z_score) / alpha

            elif distribution == 't':
                # Student's t-distribution fitting (handles fat tails better)
                from scipy.stats import t as t_dist
                df, loc, scale = t_dist.fit(returns)
                t_score = t_dist.ppf(1 - confidence, df)
                var = loc * time_horizon + scale * np.sqrt(time_horizon) * t_score
                cvar = var  # Simplified representation for t-dist expected shortfall

            else:
                raise ValueError(f"Unsupported distribution model: {distribution}")

            return {
                'var': float(var),
                'cvar': float(cvar),
                'confidence': confidence,
                'time_horizon': time_horizon,
                'method': f'parametric_{distribution}',
                'mu': float(mu),
                'sigma': float(sigma)
            }

        except Exception as e:
            self.logger.error(f"Parametric VaR calculation failure: {e}")
            return {'error': str(e)}

    def calculate_var_monte_carlo(self,
                                 returns: pd.Series,
                                 confidence: float = 0.95,
                                 time_horizon: int = 1,
                                 n_simulations: int = 10000) -> Dict[str, float]:
        """
        Monte Carlo VaR simulation via bootstrap sampling.

        Args:
            returns: Historical return series.
            confidence: Confidence level.
            time_horizon: Forecast horizon.
            n_simulations: Total number of iterations.
        """
        try:
            if len(returns) < 30:
                raise RiskManagementError("Insufficient data for Monte Carlo sampling")

            # Bootstrap sampling with replacement
            seed = self.random_seed if self.random_seed is not None else None
            simulated_returns = []
            rng = np.random.default_rng(seed)
            for _ in range(n_simulations):
                sample = rng.choice(returns.values, size=time_horizon, replace=True)
                portfolio_return = np.prod(1 + sample) - 1
                simulated_returns.append(portfolio_return)

            simulated_returns = np.array(simulated_returns)
            var = np.percentile(simulated_returns, (1 - confidence) * 100)

            # Expected Shortfall from simulation results
            tail_returns = simulated_returns[simulated_returns <= var]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var

            return {
                'var': float(var),
                'cvar': float(cvar),
                'confidence': confidence,
                'time_horizon': time_horizon,
                'method': 'monte_carlo',
                'n_simulations': n_simulations,
                'mean_simulated': float(np.mean(simulated_returns)),
                'std_simulated': float(np.std(simulated_returns))
            }

        except Exception as e:
            self.logger.error(f"Monte Carlo VaR simulation failure: {e}")
            return {'error': str(e)}

class StressTestingFramework:
    """
    Stress Testing Framework for high-impact market scenario analysis.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """Initializes the framework with predefined stress scenarios."""
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("StressTesting")

        # Standard market stress scenarios
        self.scenarios = {
            'market_crash': {'shock': -0.15, 'description': 'Sudden 15% broad market crash'},
            'volatility_spike': {'volatility_multiplier': 3.0, 'description': 'Extreme 3x volatility expansion'},
            'liquidity_crisis': {'liquidity_dryup': 0.8, 'description': 'Severe 80% market liquidity reduction'},
            'interest_rate_shock': {'rate_change': 0.025, 'description': 'Unexpected 2.5% interest rate hike'},
            'correlated_crash': {'correlation_increase': 0.8, 'description': 'Extreme asset correlation breakdown'}
        }

    def run_stress_test(self,
                       portfolio: Dict[str, float],
                       _historical_data: pd.DataFrame,
                       scenario: str = 'market_crash') -> Dict[str, Any]:
        """
        Runs a stress test simulation for the provided portfolio weights.

        Args:
            portfolio: Dictionary mapping {ticker: weight}.
            historical_data: Historical price/return dataframe.
            scenario: Predifined scenario key.

        Returns:
            Impact analysis and corrective recommendations.
        """
        try:
            if scenario not in self.scenarios:
                raise ValueError(f"Unknown stress scenario: {scenario}")

            scenario_config = self.scenarios[scenario]

            # Calculate theoretical impact on the portfolio
            if scenario == 'market_crash':
                shock = scenario_config['shock']
                portfolio_impact = sum(weight * shock for weight in portfolio.values())

            elif scenario == 'volatility_spike':
                # Model volatility expansion impact
                vol_multiplier = scenario_config['volatility_multiplier']
                # Simplified projection - scaling exposure-based VaR
                base_var = 0.02  # Baseline assumption
                stressed_var = base_var * vol_multiplier
                portfolio_impact = -stressed_var * 2  # Conservative projection

            elif scenario == 'liquidity_crisis':
                # Model liquidity premium/impact impact
                liquidity_reduction = scenario_config['liquidity_dryup']
                portfolio_impact = -0.05 * liquidity_reduction  # 5% impact per 10% liquidity loss

            else:
                portfolio_impact = -0.05  # Generic 5% fallback loss

            # Estimate recovery time under stress conditions
            recovery_days = abs(portfolio_impact) * 100 

            return {
                'scenario': scenario,
                'description': scenario_config['description'],
                'portfolio_impact': float(portfolio_impact),
                'portfolio_loss_pct': float(abs(portfolio_impact) * 100),
                'estimated_recovery_days': int(recovery_days),
                'breaches_limits': abs(portfolio_impact) > 0.1,  # 10% critical loss threshold
                'recommendations': self._generate_recommendations(portfolio_impact, scenario)
            }

        except Exception as e:
            self.logger.error(f"Stress test execution failure: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, impact: float, scenario: str) -> List[str]:
        """Generates actionable advice based on stress test outcomes."""
        recommendations = []

        if abs(impact) > 0.1:  # Critical loss (>10%)
            recommendations.append("Critical: Consider partial position liquidation to preserve capital")
            recommendations.append("Recommended: Tighten stop-loss levels across the portfolio")

        if scenario == 'market_crash':
            recommendations.append("Action: Consider hedging via options or inverse ETFs")
            recommendations.append("Action: Re-evaluate exposure to momentum-heavy assets")

        elif scenario == 'volatility_spike':
            recommendations.append("Action: Reduce position sizes to lower overall portfolio volatility")
            recommendations.append("Action: Implement volatility-targeting strategies")

        return recommendations

class LiquidityRiskAssessor:
    """
    Assesses liquidity profile of assets and estimates transaction costs/slippage.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """Initializes the assessor for liquidity modeling."""
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("LiquidityRisk")

    def assess_liquidity_risk(self,
                            ticker: str,
                            volume_data: pd.Series,
                            price_data: pd.Series,
                            position_size: float) -> Dict[str, Any]:
        """
        Evaluates an asset's liquidity footprint.

        Args:
            ticker: Asset identifier.
            volume_data: Series of historical trading volumes.
            price_data: Series of historical prices.
            position_size: Planned position size in USD.

        Returns:
            Liquidity metrics and entry/exit safety scores.
        """
        try:
            # Average daily volume calculation
            avg_daily_volume = volume_data.mean()
            avg_daily_volume_dollars = (volume_data * price_data).mean()

            # Bid-ask spread proxy (simplified via volatility)
            returns = price_data.pct_change()
            volatility = returns.std()
            spread_estimate = volatility * 0.01 

            # Market impact estimation (cost of entry/exit relative to daily turnover)
            market_impact_pct = min(position_size / avg_daily_volume_dollars, 0.1)

            # Liquidity score (0-100 scale, higher is better)
            volume_score = min(avg_daily_volume / 1000000, 1.0)  # Normalized to $1M benchmark
            spread_score = max(0, 1 - spread_estimate * 100)     # Low spread increases the score
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3) * 100

            # Qualitative risk assessment
            if liquidity_score < 30:
                risk_level = "HIGH"
                risk_description = "High liquidity risk - avoid large positions due to slippage risk"
            elif liquidity_score < 60:
                risk_level = "MEDIUM"
                risk_description = "Moderate liquidity risk - limit position sizes relative to volume"
            else:
                risk_level = "LOW"
                risk_description = "Low liquidity risk - asset is suitable for standard trading sizes"

            return {
                'ticker': ticker,
                'liquidity_score': float(liquidity_score),
                'risk_level': risk_level,
                'risk_description': risk_description,
                'avg_daily_volume': float(avg_daily_volume),
                'avg_daily_volume_dollars': float(avg_daily_volume_dollars),
                'estimated_spread_pct': float(spread_estimate),
                'market_impact_pct': float(market_impact_pct),
                'max_position_size': float(avg_daily_volume_dollars * 0.01),  # Recommended 1% threshold
                'recommendations': self._generate_liquidity_recommendations(risk_level, position_size, avg_daily_volume_dollars)
            }

        except Exception as e:
            self.logger.error(f"Liquidity risk measurement failure: {e}")
            return {'error': str(e)}

    def _generate_liquidity_recommendations(self,
                                          risk_level: str,
                                          position_size: float,
                                          avg_daily_volume: float) -> List[str]:
        """Generates specific advice for handling illiquid assets."""
        recommendations = []

        if risk_level == "HIGH":
            recommendations.append("Avoid this asset or use extremely small position sizes")
            recommendations.append("Consider more liquid alternatives in the same asset class")

        elif risk_level == "MEDIUM":
            max_safe_size = avg_daily_volume * 0.005  # Safer 0.5% threshold
            if position_size > max_safe_size:
                recommendations.append(f"Reduce position below ${max_safe_size:,.0f} to ensure safe execution")

        return recommendations

class RiskLimitsManager:
    """
    Manages and enforces hard and soft risk limits for the portfolio.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """Initializes limits from global configuration."""
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("RiskLimits")

        # Load limits from configuration or set production defaults
        risk_config = self.config.get('strategy.risk_management', {})
        self.limits = {
            'max_portfolio_var': risk_config.get('max_portfolio_var_pct', 0.05),
            'max_single_position': risk_config.get('max_single_position_pct', 0.10),
            'max_daily_loss': risk_config.get('max_daily_loss_pct', 0.03),
            'max_drawdown': risk_config.get('max_drawdown_pct', 0.15),
            'max_leverage': risk_config.get('max_leverage', 2.0),
        }

    def check_limits(self,
                    portfolio_value: float,
                    positions: Dict[str, Dict[str, Any]],
                    daily_pnl: float,
                    current_drawdown: float) -> Dict[str, Any]:
        """
        Verifies if current portfolio status adheres to global risk limits.

        Args:
            portfolio_value: Total current portfolio valuation in USD.
            positions: Dictionary of positions with size/value metrics.
            daily_pnl: Real-time daily Profit & Loss.
            current_drawdown: Portfolio peak-to-trough decline.

        Returns:
            Detailed report including any protocol violations or warnings.
        """
        violations = []
        warnings = []

        # Portfolio VaR limit check (simplified assumption-based VaR)
        estimated_var = portfolio_value * 0.02 
        if estimated_var > portfolio_value * self.limits['max_portfolio_var']:
            violations.append({
                'type': 'portfolio_var',
                'current': estimated_var / portfolio_value,
                'limit': self.limits['max_portfolio_var'],
                'message': f"Portfolio VaR {estimated_var/portfolio_value:.1%} exceeds limit of {self.limits['max_portfolio_var']:.1%}"
            })

        # Individual position concentration check
        for ticker, pos_data in positions.items():
            concentration = pos_data['value'] / portfolio_value
            if concentration > self.limits['max_single_position']:
                violations.append({
                    'type': 'single_position',
                    'ticker': ticker,
                    'current': concentration,
                    'limit': self.limits['max_single_position'],
                    'message': f"Position {ticker} concentration ({concentration:.1%}) exceeds limit of {self.limits['max_single_position']:.1%}"
                })

        # Daily loss/stop-loss check
        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        if daily_loss_pct > self.limits['max_daily_loss']:
            violations.append({
                'type': 'daily_loss',
                'current': daily_loss_pct,
                'limit': self.limits['max_daily_loss'],
                'message': f"Daily drawdown {daily_loss_pct:.1%} exceeds critical loss limit of {self.limits['max_daily_loss']:.1%}"
            })

        # Total drawdown check
        if current_drawdown > self.limits['max_drawdown']:
            violations.append({
                'type': 'drawdown',
                'current': current_drawdown,
                'limit': self.limits['max_drawdown'],
                'message': f"Total drawdown ({current_drawdown:.1%}) exceeds absolute limit of {self.limits['max_drawdown']:.1%}"
            })

        # Proactive warnings for limit approaches
        if estimated_var / portfolio_value > self.limits['max_portfolio_var'] * 0.8:
            warnings.append("Portfolio VaR is approaching critical risk threshold")

        if daily_loss_pct > self.limits['max_daily_loss'] * 0.7:
            warnings.append("Daily P&L is approaching the mandatory stop-loss threshold")

        return {
            'limits_respected': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'checked_at': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'positions_count': len(positions)
        }

class RiskManagementFramework:
    """
    Main Risk Management Framework integrating all monitoring and calculation subsystems.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """Initializes the integrated risk management stack."""
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("RiskFramework")

        # Initialize internal components
        self.var_calculator = VaRCalculator(self.config)
        self.stress_tester = StressTestingFramework(self.config)
        self.liquidity_assessor = LiquidityRiskAssessor(self.config)
        self.limits_manager = RiskLimitsManager(self.config)

        self.logger.info("Risk Management Framework initialized successfully")

    def comprehensive_risk_assessment(self,
                                    portfolio: Dict[str, float],
                                    historical_data: pd.DataFrame,
                                    current_positions: Dict[str, Dict[str, Any]],
                                    portfolio_value: float,
                                    daily_pnl: float = 0.0,
                                    current_drawdown: float = 0.0) -> Dict[str, Any]:
        """
        Executes an end-to-end holistic risk audit of the current portfolio.

        Args:
            portfolio: Dictionary of requested asset weights.
            historical_data: Market data for VaR/Stress modeling.
            current_positions: Actual live positions.
            portfolio_value: Total USD valuation.
            daily_pnl: Today's realized/unrealized P&L.
            current_drawdown: History of portfolio decline.

        Returns:
            Integrated risk report with metrics, scenario results, and alerts.
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'risk_metrics': {},
                'stress_tests': {},
                'liquidity_analysis': {},
                'limits_check': {},
                'recommendations': [],
                'alerts': []
            }

            # 1. Multi-confidence VaR modeling
            portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_data)

            if len(portfolio_returns) > 0:
                report['risk_metrics']['portfolio_var'] = self.var_calculator.calculate_var_historical(
                    portfolio_returns, confidence=0.95
                )
                report['risk_metrics']['portfolio_var_99'] = self.var_calculator.calculate_var_historical(
                    portfolio_returns, confidence=0.99
                )

            # 2. Crisis scenario modeling (Stress Testing)
            for scenario in ['market_crash', 'volatility_spike', 'liquidity_crisis']:
                report['stress_tests'][scenario] = self.stress_tester.run_stress_test(
                    portfolio, historical_data, scenario
                )

            # 3. Market execution feasibility modeling (Liquidity analysis)
            for ticker in portfolio.keys():
                if ticker in historical_data.columns:
                    ticker_data = historical_data[ticker]
                    position_value = portfolio_value * portfolio[ticker]
                    report['liquidity_analysis'][ticker] = self.liquidity_assessor.assess_liquidity_risk(
                        ticker, ticker_data, ticker_data, position_value
                    )

            # 4. Mandatory Risk Protocol validation (Limits check)
            report['limits_check'] = self.limits_manager.check_limits(
                portfolio_value, current_positions, daily_pnl, current_drawdown
            )

            # 5. Synthesis of audit results into actionable advice
            report['recommendations'] = self._generate_comprehensive_recommendations(report)
            report['alerts'] = self._generate_alerts(report)

            self.logger.info("Comprehensive portfolio risk audit completed")
            return report

        except Exception as e:
            self.logger.error(f"Comprehensive risk assessment failure: {e}")
            return {'error': str(e)}

    def _calculate_portfolio_returns(self, portfolio: Dict[str, float], data: pd.DataFrame) -> pd.Series:
        """Derives synthetic portfolio historical returns using provided weights."""
        try:
            portfolio_returns = pd.Series(0.0, index=data.index)

            for ticker, weight in portfolio.items():
                if ticker in data.columns:
                    asset_returns = data[ticker].pct_change()
                    portfolio_returns += asset_returns * weight

            return portfolio_returns.dropna()

        except Exception as e:
            self.logger.error(f"Portfolio return reconstruction failure: {e}")
            return pd.Series()

    def _generate_comprehensive_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Synthesizes all audit layers into a final list of recommendations."""
        recommendations = []

            # VaR enforcement
        if 'portfolio_var' in report['risk_metrics']:
            var_95 = report['risk_metrics']['portfolio_var'].get('var', 0)
            if var_95 < -0.05:  # High risk threshold (>5% tail loss)
                recommendations.append("High VaR warning: Consider de-risking the overall portfolio")

        # Stress test results
        for scenario, result in report['stress_tests'].items():
            if result.get('breaches_limits', False):
                recommendations.append(f"Protocol breach under {scenario} scenario: Defensive reallocation required")

        # Liquidity constraints
        for ticker, analysis in report['liquidity_analysis'].items():
            if analysis.get('risk_level') == 'HIGH':
                recommendations.append(f"Critical execution risk for {ticker}: Mandatory position reduction")

        # Hard limit violations
        if not report['limits_check'].get('limits_respected', True):
            recommendations.append("Risk limit violation: Immediate position correction mandated by safety protocol")

        return recommendations

    def _generate_alerts(self, report: Dict[str, Any]) -> List[str]:
        """Filters high-priority alerts for external notification systems."""
        alerts = []

        if not report['limits_check'].get('limits_respected', True):
            alerts.append("CRITICAL: Risk Limit Violation Detected!")

        if 'portfolio_var' in report['risk_metrics']:
            var_99 = report['risk_metrics']['portfolio_var_99'].get('var', 0)
            if var_99 < -0.10:  # Rare tail loss > 10%
                alerts.append("CRITICAL: Extreme Tail Risk Modeling (>10% loss at 99% confidence level)!")

        return alerts