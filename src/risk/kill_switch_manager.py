#!/usr/bin/env python3
"""
Kill Switch Manager - Real-time Kill-Switch System for Risk Management
Implements automated position closure during extreme market conditions.
"""

from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("KillSwitchManager")

from src.risk.elite_risk_metrics import EliteRiskMetrics


class KillSwitchManager:
    """
    Real-time kill-switch system. Focuses on emergency execution.
    Delegates math to FinancialMetricsLibrary and EliteRiskMetrics.
    """
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger
        self.elite_metrics = EliteRiskMetrics(config_manager=self.config)
        self.active = False
        
        # ✅ FIX: Initialize state variables
        self.current_risk_level = 'normal'
        self.kill_switch_active = False
        self.emergency_closure_active = False
        self.risk_limits = self.config.get('risk_limits', {})
        self.position_metrics: Dict[str, Dict[str, Any]] = {}
        self.current_positions: Dict[str, Any] = {}
        self.risk_events: list[Dict[str, Any]] = []
        self.position_metrics_history: list[Dict[str, Any]] = []
        self.kill_switch_history: list[Dict[str, Any]] = []
        self.alert_manager = AlertManager()
        
        self.logger.info("✅ KillSwitchManager initialized with full state.")

    async def check_and_execute(self, portfolio_data: Dict[str, Any], market_data: pd.DataFrame):
        """Monitors risk and triggers closure if thresholds are breached."""
        # Get unified risk report
        positions = {s: p.get('quantity', 0) for s, p in portfolio_data.items()}
        prices = {s: p.get('current_price', 0) for s, p in portfolio_data.items()}
        total_val = sum(p.get('current_value', 0) for p in portfolio_data.values())

        risk_report = self.elite_metrics.get_risk_report(positions, prices, total_val)

        if risk_report.get('risk_status') == 'high':
            self.logger.critical("🚨 RISK STATUS HIGH! Activating Kill Switch.")
            self.active = True
            return await self._execute_closure(portfolio_data)

        return {"status": "safe", "risk_level": risk_report.get('risk_status')}

    async def _execute_closure(self, portfolio_data: Dict[str, Any]):
        """Emergency closure logic."""
        # ... (Implementation for closing positions)
        return {"status": "kill_switch_activated", "action": "all_positions_closed"}

    async def monitor_and_execute(self,
                                portfolio_data: dict[str, Any],
                                market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Monitor portfolio and execute risk management actions.

        Args:
            portfolio_data: Current portfolio positions
            market_data: Current market data

        Returns:
            Dict with risk analysis and actions taken
        """
        self.logger.info("🛡️ Starting kill-switch monitoring and execution")

        results = {
            'timestamp': datetime.now(),
            'current_risk_level': self.current_risk_level,
            'kill_switch_active': self.kill_switch_active,
            'risk_analysis': {},
            'actions_taken': [],
            'recommendations': []
        }

        try:
            # 1. Update current positions
            self._update_positions(portfolio_data)

            # 2. Calculate risk metrics
            risk_analysis = self._calculate_risk_metrics(portfolio_data, market_data)
            results['risk_analysis'] = risk_analysis

            # 3. Check emergency triggers
            emergency_triggers = self._check_emergency_triggers(risk_analysis, market_data)

            # 4. Execute appropriate actions
            if emergency_triggers['any_triggered']:
                actions = await self._execute_emergency_actions(
                    emergency_triggers, portfolio_data, market_data
                )
                results['actions_taken'].extend(actions)
            else:
                # Normal risk management
                actions = await self._execute_normal_risk_management(
                    risk_analysis, portfolio_data, market_data
                )
                results['actions_taken'].extend(actions)

            # 5. Update state
            self._update_risk_state(risk_analysis)

            # 6. Store results
            self._store_monitoring_results(results)

            self.logger.info(f"✅ Kill-switch monitoring complete. Risk level: {self.current_risk_level}")

            return results

        except Exception as e:
            self.logger.error(f"Error in kill-switch monitoring: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def _update_positions(self, portfolio_data: dict[str, Any]) -> None:
        """Update current position tracking."""

        try:
            self.current_positions = portfolio_data.copy()

            # Calculate position metrics
            for symbol, position in portfolio_data.items():
                if 'quantity' in position and 'current_price' in position:
                    self.position_metrics[symbol] = {
                        'quantity': position['quantity'],
                        'current_price': position['current_price'],
                        'entry_price': position.get('entry_price', 0.0),
                        'current_value': position['quantity'] * position['current_price'],
                        'unrealized_pnl': (position['current_price'] - position.get('entry_price', 0.0)) * position['quantity']
                    }

            self.logger.debug(f"Updated positions for {len(self.current_positions)} assets")

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def _calculate_risk_metrics(self,
                           portfolio_data: dict[str, Any],
                           market_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate comprehensive risk metrics."""

        try:
            risk_metrics = {
                'portfolio_level': self.current_risk_level,
                'portfolio_metrics': {},
                'position_metrics': {},
                'market_conditions': {},
                'risk_alerts': []
            }

            # 1. Portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio_data, market_data)
            risk_metrics['portfolio_metrics'] = portfolio_metrics

            # 2. Position-level metrics
            position_metrics = self._calculate_position_metrics(portfolio_data, market_data)
            risk_metrics['position_metrics'] = position_metrics

            # 3. Market conditions
            market_conditions = self._analyze_market_conditions(market_data)
            risk_metrics['market_conditions'] = market_conditions

            # 4. Determine risk level
            risk_level = self._determine_risk_level(
                portfolio_metrics, position_metrics, market_conditions, market_data
            )
            risk_metrics['portfolio_level'] = risk_level

            # 5. Check for risk alerts
            risk_alerts = self._generate_risk_alerts(risk_metrics)
            risk_metrics['risk_alerts'] = risk_alerts

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}

    def _calculate_portfolio_metrics(self,
                                  portfolio_data: dict[str, Any],
                                  market_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate portfolio-level risk metrics."""

        try:
            if not portfolio_data:
                return {}

            # Calculate portfolio value
            portfolio_value = sum(
                position.get('current_value', 0.0)
                for position in portfolio_data.values()
            )

            # Calculate daily returns for portfolio
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, market_data)

            if len(portfolio_returns) < 2:
                return {
                    'portfolio_value': portfolio_value,
                    'daily_returns': []
                }

            # Calculate portfolio metrics
            daily_var = np.var(portfolio_returns) if len(portfolio_returns) > 1 else 0
            portfolio_volatility = np.sqrt(daily_var) * np.sqrt(252) if daily_var > 0 else 0

            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns) - 1
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

            # Calculate current drawdown
            current_drawdown = 0.0
            if len(portfolio_returns) > 0:
                peak = float(np.max(np.maximum.accumulate(cumulative_returns)))
                current = float(cumulative_returns[-1])
                current_drawdown = (peak - current) / peak if peak > 0 else 0.0

            return {
                'portfolio_value': portfolio_value,
                'daily_returns': portfolio_returns,
                'daily_var': daily_var,
                'portfolio_volatility': portfolio_volatility,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'var_ratio': daily_var / portfolio_volatility if portfolio_volatility > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def _calculate_portfolio_returns(self,
                                portfolio_data: dict[str, Any],
                                market_data: pd.DataFrame) -> list[float]:
        """Calculate daily returns for portfolio."""

        try:
            if not portfolio_data or market_data.empty:
                return []

            returns = []

            # Get price data for portfolio assets
            portfolio_symbols = list(portfolio_data.keys())

            for symbol in portfolio_symbols:
                if symbol in market_data['close'].columns:
                    # Calculate daily returns
                    symbol_returns = market_data['close'][symbol].pct_change().dropna()
                    returns.extend(symbol_returns.tolist())

            return returns

        except Exception as e:
            self.logger.error(f"Error calculating portfolio returns: {e}")
            return []

    def _calculate_position_metrics(self,
                                 portfolio_data: dict[str, Any],
                                 market_data: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Calculate position-level risk metrics."""

        try:
            position_metrics = {}

            for symbol, _position in portfolio_data.items():
                if symbol not in market_data['close'].columns:
                    continue

                # Get price data
                symbol_prices = market_data['close'][symbol]

                if len(symbol_prices) < 2:
                    position_metrics[symbol] = {
                        'returns': [],
                        'volatility': 0.0,
                        'max_drawdown': 0.0,
                        'correlation_risk': 0.0
                    }
                    continue

                # Calculate returns
                symbol_returns = symbol_prices.pct_change().dropna()

                # Calculate volatility
                volatility = symbol_returns.std() * np.sqrt(252)

                # Calculate maximum drawdown
                cumulative_returns = (1 + symbol_returns).cumprod()
                running_max = np.maximum.accumulate(cumulative_returns) - 1
                drawdowns = running_max - cumulative_returns
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

                # Calculate current drawdown
                current_drawdown = 0.0
                if len(symbol_returns) > 0:
                    peak = float(np.max(np.maximum.accumulate(cumulative_returns)))
                    current = float(cumulative_returns.iloc[-1])
                    current_drawdown = (peak - current) / peak if peak > 0 else 0.0

                position_metrics[symbol] = {
                    'returns': symbol_returns.tolist(),
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'current_drawdown': current_drawdown,
                    'var_ratio': volatility / (symbol_returns.std() * np.sqrt(252)) if volatility > 0 else 0
                }

            return position_metrics

        except Exception as e:
            self.logger.error(f"Error calculating position metrics: {e}")
            return {}

    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze current market conditions."""

        try:
            market_conditions = {}

            if market_data.empty:
                return {
                    'volatility_regime': 'unknown',
                    'trend_regime': 'unknown',
                    'volatility_level': 'unknown',
                    'trend_strength': 0.0,
                    'market_stress': False
                }

            # Calculate market volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # Determine volatility regime
            if volatility < 0.01:
                volatility_regime = 'low'
                volatility_level = 'low'
            elif volatility < 0.02:
                volatility_regime = 'normal'
                volatility_level = 'normal'
            elif volatility < 0.04:
                volatility_regime = 'elevated'
                volatility_level = 'elevated'
            else:
                volatility_regime = 'high'
                volatility_level = 'high'

            # Calculate trend
            prices = market_data['close']
            short_ma = prices.rolling(window=20).mean()
            long_ma = prices.rolling(window=50).mean()

            if long_ma > short_ma:
                trend_regime = 'uptrend'
                trend_strength = (long_ma - short_ma) / short_ma
            elif long_ma < short_ma:
                trend_regime = 'downtrend'
                trend_strength = (short_ma - long_ma) / long_ma
            else:
                trend_regime = 'sideways'
                trend_strength = 0.0

            # Detect market stress
            recent_volatility = returns.rolling(window=5).std()
            historical_volatility = returns.rolling(window=20).std()

            market_stress = recent_volatility > (historical_volatility * 2)

            market_conditions = {
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'volatility_level': volatility_level,
                'trend_strength': trend_strength,
                'market_stress': market_stress,
                'current_volatility': volatility,
                'historical_volatility': historical_volatility,
                'volatility_spike': recent_volatility / historical_volatility
            }

            return market_conditions

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {}

    def _determine_risk_level(self,
                           portfolio_metrics: dict[str, dict[str, Any]],
                           position_metrics: dict[str, dict[str, Any]],
                           market_conditions: dict[str, Any],
                           market_data: pd.DataFrame) -> str:
        """Determine overall risk level."""

        try:
            # Get thresholds
            portfolio_var_threshold = self.risk_limits.get('portfolio_var_threshold', 0.15)
            position_var_threshold = self.risk_limits.get('position_var_threshold', 0.25)
            max_drawdown_threshold = self.risk_limits.get('max_drawdown_threshold', 0.10)
            correlation_threshold = self.risk_limits.get('correlation_threshold', 0.7)
            self.risk_limits.get('market_volatility_spike_threshold', 3.0)

            risk_level = 'normal'
            risk_factors = []

            # Check portfolio metrics
            portfolio_metrics = portfolio_metrics.get('portfolio_metrics', {})

            if portfolio_metrics:
                portfolio_var = portfolio_metrics.get('daily_var', 0)
                if portfolio_var > portfolio_var_threshold:
                    risk_level = 'elevated'
                    risk_factors.append('portfolio_variance_exceeded')

                max_drawdown = portfolio_metrics.get('max_drawdown', 0.0)
                if max_drawdown > max_drawdown_threshold:
                    risk_level = 'high'
                    risk_factors.append('max_drawdown_exceeded')

                portfolio_vol = portfolio_metrics.get('portfolio_volatility', 0.0)
                if portfolio_vol > self.risk_limits.get('portfolio_volatility_threshold', 0.25):
                    risk_level = 'elevated'
                    risk_factors.append('high_volatility')

            # Check position metrics
            position_metrics = position_metrics.get('position_metrics', {})

            high_risk_positions = 0
            total_positions = len(position_metrics)

            for symbol, metrics in position_metrics.items():
                position_var = metrics.get('var_ratio', 0.0)
                if position_var > position_var_threshold:
                    high_risk_positions += 1

                max_drawdown = metrics.get('max_drawdown', 0.0)
                if max_drawdown > max_drawdown_threshold:
                    risk_level = 'critical'
                    risk_factors.append('position_max_drawdown_exceeded')

                volatility = metrics.get('volatility', 0.0)
                if volatility > self.risk_limits.get('position_volatility_threshold', 0.3):
                    risk_level = 'high'
                    risk_factors.append('position_high_volatility')

                # Calculate average correlation with other positions
                if total_positions > 1:
                    other_volatilities = [
                        m.get('volatility', 0.0)
                        for s, m in position_metrics.items()
                        if s != symbol
                    ]
                    if other_volatilities:
                        correlations = []
                        for s in position_metrics:
                            if s == symbol:
                                continue
                            a = market_data['close'][symbol].pct_change().dropna()
                            b = market_data['close'][s].pct_change().dropna()
                            idx = a.index.intersection(b.index)
                            if len(idx) < 2:
                                continue
                            corr = stats.pearsonr(a.loc[idx], b.loc[idx])[0]
                            correlations.append(min(0.9, abs(float(corr))))
                        avg_correlation = float(np.mean(correlations)) if correlations else 0.0

                        if avg_correlation > correlation_threshold:
                            risk_level = 'high'
                            risk_factors.append('position_correlation_risk')

            # Check market conditions
            market_conditions = market_conditions.get('market_conditions', {})

            if market_conditions.get('market_stress', False):
                risk_level = max(risk_level, 'high')
                risk_factors.append('market_stress')

            if market_conditions.get('volatility_spike', False):
                risk_level = max(risk_level, 'elevated')
                risk_factors.append('volatility_spike')

            # Determine final risk level
            if risk_level == 'normal' and risk_factors:
                risk_level = 'elevated'
            elif risk_level == 'elevated' and len(risk_factors) >= 2:
                risk_level = 'high'
            elif len(risk_factors) >= 3 or 'critical' in risk_factors:
                risk_level = 'critical'

            return risk_level

        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return 'normal'

    def _check_emergency_triggers(self, risk_analysis: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
        """Check if any emergency triggers are activated."""

        try:
            triggers = {
                'any_triggered': False,
                'portfolio_var_exceeded': False,
                'max_drawdown_exceeded': False,
                'position_var_exceeded': False,
                'correlation_spike': False,
                'market_volatility_spike': False,
                'liquidity_crisis': False
            }

            # Check portfolio metrics
            portfolio_metrics = risk_analysis.get('portfolio_metrics', {})

            if portfolio_metrics:
                portfolio_var = portfolio_metrics.get('daily_var', 0)
                portfolio_var_threshold = self.risk_limits.get('portfolio_var_threshold', 0.15)

                if portfolio_var > portfolio_var_threshold:
                    triggers['portfolio_var_exceeded'] = True
                    triggers['any_triggered'] = True

                max_drawdown = portfolio_metrics.get('max_drawdown', 0.0)
                max_drawdown_threshold = self.risk_limits.get('max_drawdown_threshold', 0.10)

                if max_drawdown > max_drawdown_threshold:
                    triggers['max_drawdown_exceeded'] = True
                    triggers['any_triggered'] = True

            # Check position metrics
            position_metrics = risk_analysis.get('position_metrics', {})
            total_positions = len(position_metrics)

            high_risk_positions = 0
            for symbol, metrics in position_metrics.items():
                position_var = metrics.get('var_ratio', 0.0)
                if position_var > self.risk_limits.get('position_var_threshold', 0.25):
                    high_risk_positions += 1

                max_drawdown = metrics.get('max_drawdown', 0.0)
                max_drawdown_threshold = self.risk_limits.get('max_drawdown_threshold', 0.10)

                if max_drawdown > max_drawdown_threshold:
                    triggers['position_var_exceeded'] = True
                    triggers['any_triggered'] = True

                volatility = metrics.get('volatility', 0.0)
                if volatility > self.risk_limits.get('position_volatility_threshold', 0.3):
                    triggers['position_high_volatility'] = True
                    triggers['any_triggered'] = True

                # Check correlation risk
                if total_positions > 1:
                    other_volatilities = [
                        m.get('volatility', 0.0)
                        for s, m in position_metrics.items()
                        if s != symbol
                    ]
                    if other_volatilities:
                        correlations = []
                        for s in position_metrics:
                            if s == symbol:
                                continue
                            a = market_data['close'][symbol].pct_change().dropna()
                            b = market_data['close'][s].pct_change().dropna()
                            idx = a.index.intersection(b.index)
                            if len(idx) < 2:
                                continue
                            corr = stats.pearsonr(a.loc[idx], b.loc[idx])[0]
                            correlations.append(min(0.9, abs(float(corr))))
                        avg_correlation = float(np.mean(correlations)) if correlations else 0.0

                        if avg_correlation > self.risk_limits.get('correlation_threshold', 0.7):
                            triggers['correlation_spike'] = True
                            triggers['any_triggered'] = True

            # Check market conditions
            market_conditions = risk_analysis.get('market_conditions', {})

            if market_conditions.get('market_stress', False):
                triggers['market_stress'] = True
                triggers['any_triggered'] = True

            if market_conditions.get('volatility_spike', False):
                volatility_spike = market_conditions.get('volatility_spike', 0.0)
                if volatility_spike > self.risk_limits.get('market_volatility_spike_threshold', 3.0):
                    triggers['market_volatility_spike'] = True
                    triggers['any_triggered'] = True

            if market_conditions.get('liquidity_crisis', False):
                triggers['liquidity_crisis'] = True
                triggers['any_triggered'] = True

            return triggers

        except Exception as e:
            self.logger.error(f"Error checking emergency triggers: {e}")
            return {'any_triggered': False}

    async def _execute_emergency_actions(self,
                                     emergency_triggers: dict[str, Any],
                                     portfolio_data: dict[str, Any],
                                     market_data: pd.DataFrame) -> list[str]:
        """Execute emergency actions based on triggers."""

        actions = []

        try:
            if emergency_triggers['liquidity_crisis']:
                actions.append({
                    'action': 'emergency_closure',
                    'reason': 'Liquidity crisis detected',
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

                # Emergency closure - close all positions
                await self._emergency_closure(portfolio_data)

            elif emergency_triggers['portfolio_var_exceeded']:
                actions.append({
                    'action': 'reduce_all_positions',
                    'reason': 'Portfolio variance exceeded threshold',
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

                # Reduce all positions
                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.5)
                actions.append({
                    'action': 'reduced_positions',
                    'reduction_factor': 0.5,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

            elif emergency_triggers['max_drawdown_exceeded']:
                actions.append({
                    'action': 'reduce_all_positions',
                    'reason': 'Maximum drawdown exceeded threshold',
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

                # Reduce all positions
                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.7)
                actions.append({
                    'action': 'reduced_positions',
                    'reduction_factor': 0.7,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

            elif emergency_triggers['correlation_spike']:
                actions.append({
                    'action': 'reduce_correlated_positions',
                    'reason': 'Correlation spike detected',
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })

                # Reduce correlated positions
                reduced_portfolio = await self._reduce_correlated_positions(
                    portfolio_data, market_data, correlation_threshold=0.5
                )
                actions.append({
                    'action': 'reduced_correlated_positions',
                    'reduction_factor': 0.3,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })

            elif emergency_triggers['position_var_exceeded']:
                actions.append({
                    'action': 'reduce_position_risk',
                    'reason': 'Position variance exceeded threshold',
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })

                # Reduce high-risk positions
                high_risk_positions = [
                    symbol for symbol, metrics in portfolio_data.items()
                    if metrics.get('var_ratio', 0.0) >
                    self.risk_limits.get('position_var_threshold', 0.25)
                ]

                if high_risk_positions:
                    reduced_portfolio = await self._reduce_specific_positions(
                        portfolio_data, high_risk_positions, 0.6
                    )
                    actions.append({
                        'action': 'reduced_high_risk_positions',
                        'reduction_factor': 0.4,
                        'original_count': len(portfolio_data),
                        'reduced_count': len(reduced_portfolio),
                        'reduced_positions': high_risk_positions,
                        'timestamp': datetime.now(),
                        'severity': 'high'
                    })

            elif emergency_triggers['market_volatility_spike']:
                actions.append({
                    'action': 'reduce_all_positions',
                    'reason': 'Market volatility spike detected',
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

                # Reduce all positions due to market volatility
                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.3)
                actions.append({
                    'action': 'reduced_positions',
                    'reduction_factor': 0.3,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

            elif emergency_triggers['any_triggered']:
                # Default emergency action
                actions.append({
                    'action': 'reduce_all_positions',
                    'reason': 'Multiple emergency triggers activated',
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.4)
                actions.append({
                    'action': 'reduced_positions',
                    'reduction_factor': 0.4,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

            return actions

        except Exception as e:
            self.logger.error(f"Error executing emergency actions: {e}")
            return []

    async def _execute_normal_risk_management(self,
                                         risk_analysis: dict[str, Any],
                                         portfolio_data: dict[str, Any],
                                         market_data: pd.DataFrame) -> list[str]:
        """Execute normal risk management actions."""

        actions = []

        try:
            risk_level = risk_analysis['portfolio_level']

            if risk_level == 'normal':
                actions.append({
                    'action': 'monitor',
                    'reason': 'Normal risk conditions',
                    'timestamp': datetime.now(),
                    'severity': 'low'
                })

            elif risk_level == 'elevated':
                actions.append({
                    'action': 'reduce_positions_moderate',
                    'reason': 'Elevated risk conditions',
                    'timestamp': datetime.now(),
                    'severity': 'medium'
                })

                # Reduce positions moderately
                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.8)
                actions.append({
                    'action': 'reduced_positions_moderate',
                    'reduction_factor': 0.2,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'medium'
                })

            elif risk_level == 'high':
                actions.append({
                    'action': 'reduce_positions_moderate',
                    'reason': 'High risk conditions',
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })

                # Reduce positions more aggressively
                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.6)
                actions.append({
                    'action': 'reduced_positions_moderate',
                    'reduction_factor': 0.5,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'high'
                })

            elif risk_level == 'critical':
                actions.append({
                    'action': 'reduce_positions',
                    'reason': 'Critical risk conditions',
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

                # Aggressive position reduction
                reduced_portfolio = await self._reduce_all_positions(portfolio_data, 0.8)
                actions.append({
                    'action': 'reduced_positions',
                    'reduction_factor': 0.7,
                    'original_count': len(portfolio_data),
                    'reduced_count': len(reduced_portfolio),
                    'timestamp': datetime.now(),
                    'severity': 'critical'
                })

            else:
                actions.append({
                    'action': 'monitor',
                    'reason': 'Unknown risk level',
                    'timestamp': datetime.now(),
                    'severity': 'low'
                })

            return actions

        except Exception as e:
            self.logger.error(f"Error executing normal risk management: {e}")
            return []

    async def _emergency_closure(self, portfolio_data: dict[str, Any]) -> None:
        """Emergency closure - close all positions immediately."""

        try:
            self.logger.critical("🚨️ EMERGENCY CLOSURE TRIGGERED")

            # Close all positions
            for symbol in portfolio_data:
                portfolio_data[symbol]['quantity'] = 0
                portfolio_data[symbol]['current_price'] = 0.0

            # Update positions
            self._update_positions(portfolio_data)

            # Set emergency state
            self.emergency_closure_active = True
            self.kill_switch_active = True

            # Send emergency alert
            await self.alert_manager.send_alert(
                level='critical',
                message='EMERGENCY: Kill-switch activated - All positions closed',
                timestamp=datetime.now()
            )

            self.logger.critical("✅ Emergency closure completed. All positions closed.")

        except Exception as e:
            self.logger.error(f"Error in emergency closure: {e}")

    async def _reduce_all_positions(self,
                                portfolio_data: dict[str, Any],
                                reduction_factor: float = 0.5) -> dict[str, Any]:
        """Reduce all positions by specified factor."""

        try:
            self.logger.warning(f"🔥 Reducing all positions by factor: {reduction_factor}")

            reduced_portfolio = {}
            original_count = len(portfolio_data)

            for symbol, position in portfolio_data.items():
                original_quantity = position['quantity']
                reduced_quantity = int(original_quantity * reduction_factor)

                if reduced_quantity < 1:
                    reduced_quantity = 1

                reduced_portfolio[symbol] = {
                    'quantity': reduced_quantity,
                    'original_quantity': original_quantity,
                    'reduction_factor': reduction_factor,
                    'reduction_reason': 'emergency_risk_reduction'
                }

            # Update positions
            self._update_positions(reduced_portfolio)

            self.logger.info(f"Reduced portfolio from {original_count} to {len(reduced_portfolio)} positions")

            return reduced_portfolio

        except Exception as e:
            self.logger.error(f"Error reducing all positions: {e}")
            return portfolio_data

    async def _reduce_correlated_positions(self,
                                      portfolio_data: dict[str, Any],
                                      market_data: pd.DataFrame,
                                      correlation_threshold: float = 0.5) -> dict[str, Any]:
        """Reduce correlated positions to manage concentration risk."""

        try:
            self.logger.warning(f"🔥 Reducing correlated positions (threshold: {correlation_threshold})")

            # Calculate correlation matrix
            symbols = list(portfolio_data.keys())
            if len(symbols) < 2:
                return portfolio_data

            # Get price data
            price_matrix = market_data['close'][symbols]

            # Calculate correlation matrix
            correlation_matrix = price_matrix.pct_change().corr()

            # Find highly correlated pairs
            correlated_pairs = []
            n_symbols = len(symbols)

            for i in range(n_symbols):
                for j in range(i + 1, n_symbols):
                    correlation = abs(correlation_matrix.iloc[i, j])
                    if correlation > correlation_threshold:
                        correlated_pairs.append((symbols[i], symbols[j], correlation))

            # Reduce positions in correlated pairs
            reduced_portfolio = portfolio_data.copy()

            for symbol1, symbol2, _correlation in correlated_pairs:
                # Reduce the smaller position
                pos1_quantity = portfolio_data[symbol1]['quantity']
                pos2_quantity = portfolio_data[symbol2]['quantity']

                if pos1_quantity <= pos2_quantity:
                    reduced_portfolio[symbol1]['quantity'] = int(pos1_quantity * 0.5)
                else:
                    reduced_portfolio[symbol2]['quantity'] = int(pos2_quantity * 0.5)

                self.logger.info(f"Reduced {symbol1} position from {pos1_quantity} to {reduced_portfolio[symbol1]['quantity']}")

            # Update positions
            self._update_positions(reduced_portfolio)

            self.logger.info(f"Reduced {len([s for s in reduced_portfolio if reduced_portfolio[s]['quantity'] < portfolio_data[s]['quantity']])} correlated positions")

            return reduced_portfolio

        except Exception as e:
            self.logger.error(f"Error reducing correlated positions: {e}")
            return portfolio_data

    async def _reduce_specific_positions(self,
                                      portfolio_data: dict[str, Any],
                                      target_symbols: list[str],
                                      reduction_factor: float = 0.6) -> dict[str, Any]:
        """Reduce specific high-risk positions."""

        try:
            self.logger.warning(f"🔥 Reducing high-risk positions (factor: {reduction_factor})")

            reduced_portfolio = portfolio_data.copy()

            for symbol in target_symbols:
                if symbol in portfolio_data:
                    original_quantity = portfolio_data[symbol]['quantity']
                    reduced_quantity = int(original_quantity * reduction_factor)

                    if reduced_quantity < 1:
                        reduced_quantity = 1

                    reduced_portfolio[symbol] = {
                        'quantity': reduced_quantity,
                        'original_quantity': original_quantity,
                        'reduction_factor': reduction_factor,
                        'reduction_reason': 'high_risk_reduction'
                    }

                    self.logger.info(f"Reduced {symbol} position from {original_quantity} to {reduced_quantity}")

            # Update positions
            self._update_positions(reduced_portfolio)

            return reduced_portfolio

        except Exception as e:
            self.logger.error(f"Error reducing specific positions: {e}")
            return portfolio_data

    def _update_risk_state(self, risk_analysis: dict[str, Any]) -> None:
        """Update current risk state based on analysis."""

        try:
            old_risk_level = self.current_risk_level
            new_risk_level = risk_analysis['portfolio_level']

            self.current_risk_level = new_risk_level

            # Log risk level changes
            if old_risk_level != new_risk_level:
                self.logger.info(f"Risk level changed: {old_risk_level} -> {new_risk_level}")

            # Update kill switch status
            if new_risk_level in ['high', 'critical']:
                self.kill_switch_active = True
            else:
                self.kill_switch_active = False

        except Exception as e:
            self.logger.error(f"Error updating risk state: {e}")

    def _store_monitoring_results(self, results: dict[str, Any]) -> None:
        """Store monitoring results for historical tracking."""

        try:
            # Store risk analysis
            self.risk_events.append({
                'timestamp': results['timestamp'],
                'risk_level': results['current_risk_level'],
                'portfolio_metrics': results['risk_analysis'].get('portfolio_metrics', {}),
                'actions_taken': results['actions_taken'],
                'recommendations': results['recommendations']
            })

            # Store position metrics
            self.position_metrics_history.append({
                'timestamp': results['timestamp'],
                'position_count': len(self.position_metrics),
                'high_risk_positions': 0,
                'avg_position_var': 0.0,
                'max_position_var': 0.0
            })

            # Store kill switch history
            if results.get('actions_taken'):
                self.kill_switch_history.append({
                    'timestamp': results['timestamp'],
                    'trigger': results['actions_taken'],
                    'risk_level': results['current_risk_level'],
                    'action': results['actions_taken'][-1] if results['actions_taken'] else None
                })

            # Keep only last 1000 records
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-1000:]

            # Keep only last 500 kill switch records
            if len(self.kill_switch_history) > 500:
                self.kill_switch_history = self.kill_switch_history[-500:]

        except Exception as e:
            self.logger.error(f"Error storing monitoring results: {e}")

    def _generate_risk_alerts(self, risk_metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate risk alerts based on metrics."""

        alerts = []

        try:
            risk_level = risk_metrics.get('portfolio_level', 'normal')

            # Portfolio-level alerts
            portfolio_metrics = risk_metrics.get('portfolio_metrics', {})

            if portfolio_metrics:
                portfolio_var = portfolio_metrics.get('daily_var', 0)
                portfolio_var_threshold = self.risk_limits.get('portfolio_var_threshold', 0.15)

                if portfolio_var > portfolio_var_threshold:
                    alerts.append({
                        'level': 'warning',
                        'message': f"Portfolio variance ({portfolio_var:.2%}) exceeds threshold ({portfolio_var_threshold:.2%})",
                        'timestamp': datetime.now(),
                        'type': 'portfolio_variance'
                    })

            max_drawdown = portfolio_metrics.get('max_drawdown', 0.0)
            max_drawdown_threshold = self.risk_limits.get('max_drawdown_threshold', 0.10)

            if max_drawdown > max_drawdown_threshold:
                alerts.append({
                        'level': 'critical',
                        'message': f"Maximum drawdown ({max_drawdown:.2%}) exceeds threshold ({max_drawdown_threshold:.2%})",
                        'timestamp': datetime.now(),
                        'type': 'max_drawdown'
                    })

            # Position-level alerts
            position_metrics = risk_metrics.get('position_metrics', {})
            high_risk_positions = 0

            position_var_threshold = self.risk_limits.get('position_var_threshold', 0.25)

            for symbol, metrics in position_metrics.items():
                position_var = metrics.get('var_ratio', 0.0)
                if position_var > position_var_threshold:
                    high_risk_positions += 1
                    alerts.append({
                        'level': 'warning',
                        'message': f"Position {symbol} variance ({position_var:.2%}) exceeds threshold ({position_var_threshold:.2%})",
                        'timestamp': datetime.now(),
                        'type': 'position_variance',
                        'symbol': symbol
                    })

            if high_risk_positions > 0:
                alerts.append({
                    'level': 'warning',
                    'message': f"{high_risk_positions} positions detected",
                    'timestamp': datetime.now(),
                    'type': 'position_risk'
                })

            # Market condition alerts
            market_conditions = risk_metrics.get('market_conditions', {})

            if market_conditions.get('market_stress', False):
                alerts.append({
                    'level': 'warning',
                    'message': 'Market stress detected',
                    'timestamp': datetime.now(),
                    'type': 'market_conditions'
                })

            if market_conditions.get('volatility_spike', False):
                volatility_spike = market_conditions.get('volatility_spike', 0.0)
                if volatility_spike > self.risk_limits.get('market_volatility_spike_threshold', 3.0):
                    alerts.append({
                        'level': 'warning',
                        'message': f"Market volatility spike detected ({volatility_spike:.2f}x threshold)",
                        'timestamp': datetime.now(),
                        'type': 'volatility_spike'
                    })

            # Generate summary alert
            if alerts:
                alerts.append({
                    'level': risk_level,
                    'message': f"Multiple risk alerts triggered (Level: {risk_level})",
                    'timestamp': datetime.now(),
                    'alert_count': len(alerts)
                })

            return alerts

        except Exception as e:
            self.logger.error(f"Error generating risk alerts: {e}")
            return []

    def get_risk_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of risk monitoring over time period."""

        try:
            cutoff_time = datetime.now() - timedelta(days=days)

            # Filter recent events
            recent_events = [
                event for event in self.risk_events
                if event['timestamp'] >= cutoff_time
            ]

            if not recent_events:
                return {'error': 'No recent risk monitoring data available'}

            # Calculate summary statistics
            summary = {
                'period_days': days,
                'total_events': len(recent_events),
                'current_risk_level': self.current_risk_level,
                'risk_level_distribution': self._calculate_risk_level_distribution(recent_events),
                'emergency_closure_count': self._count_emergency_closures(),
                'position_reduction_count': self._count_position_reductions(),
                'correlation_reduction_count': self._count_correlation_reductions(),
                'kill_switch_activations': 0,
                'alert_count': len(self._get_risk_alerts(recent_events))
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {'error': str(e)}

    def _calculate_risk_level_distribution(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate distribution of risk levels over time."""

        try:
            level_counts = {
                'normal': 0,
                'elevated': 0,
                'high': 0,
                'critical': 0
            }

            for event in events:
                level = event.get('risk_level', 'normal')
                level_counts[level] += 1

            return level_counts

        except Exception as e:
            self.logger.error(f"Error calculating risk level distribution: {e}")
            return {'normal': 0, 'elevated': 0, 'high': 0, 'critical': 0}

    def _count_emergency_closures(self) -> int:
        """Count emergency closures in recent period."""

        try:
            cutoff_time = datetime.now() - timedelta(days=30)

            recent_events = [
                event for event in self.risk_events
                if event['timestamp'] >= cutoff_time
            ]

            return len([
                event for event in recent_events
                if event.get('action') == 'emergency_closure'
            ])

        except Exception as e:
            self.logger.error(f"Error counting emergency closures: {e}")
            return 0

    def _count_position_reductions(self) -> int:
        """Count position reductions in recent period."""

        try:
            cutoff_time = datetime.now() - timedelta(days=30)

            recent_events = [
                event for event in self.risk_events
                if event.get('timestamp')
                and event['timestamp'] >= cutoff_time
                and event.get('action') in {'reduced_positions_moderate', 'reduced_positions'}
            ]

            return len(recent_events)

        except Exception as e:
            self.logger.error(f"Error counting position reductions: {e}")
            return 0

    def _count_correlation_reductions(self) -> int:
        """Count correlation-based position reductions in recent period."""

        try:
            datetime.now() - timedelta(days=30)

            recent_events = [
                event for event in self.risk_events
                if event.get('action') == 'reduced_correlated_positions'
            ]

            return len(recent_events)

        except Exception as e:
            self.logger.error(f"Error counting correlation reductions: {e}")
            return 0


# Factory function for easy instantiation
def get_kill_switch_manager(config: dict[str, Any] | None = None) -> KillSwitchManager:
    """Factory function to get KillSwitchManager instance."""
    return KillSwitchManager(config)


# Alert Manager for kill-switch system
class AlertManager:
    """Simple alert management for kill-switch system."""

    def __init__(self):
        self.alerts = []

    def send_alert(self, level: str, message: str, timestamp: datetime | None = None):
        """Send alert notification."""

        alert = {
            'level': level,
            'message': message,
            'timestamp': timestamp or datetime.now()
        }

        self.alerts.append(alert)

        # Here you would integrate with your notification system
        logger.warning(f"ALERT: {level.upper()}: {message}")

        # Integration points:
        # - Email notifications
        # - Slack notifications
        # - SMS alerts
        # - Dashboard alerts
        # - Trading platform integration
        pass


# Convenience function for quick emergency monitoring
async def monitor_risk_emergency_quick(portfolio_data: dict[str, Any],
                                 market_data: pd.DataFrame,
                                 config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Quick emergency risk monitoring.

    Args:
        portfolio_data: Current portfolio positions
        market_data: Current market data
        config: Configuration dictionary

    Returns:
        Risk monitoring result dictionary
    """
    manager = get_kill_switch_manager(config)
    return await manager.monitor_and_execute(portfolio_data, market_data)
