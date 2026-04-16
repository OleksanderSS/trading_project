"""
Risk Management Framework - Основний модуль управління ризиками

Включає:
- Value at Risk (VaR) розрахунки
- Conditional VaR (CVaR)
- Stress Testing Framework
- Liquidity Risk Assessment
- Risk Limits Management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.stats import norm, t
from datetime import datetime, timedelta
import logging

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager

logger = ProjectLogger.get_logger("RiskManagement")

class RiskManagementError(Exception):
    """Custom exception for risk management errors"""
    pass

class VaRCalculator:
    """
    Value at Risk Calculator - різноманітні методи розрахунку VaR
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        self.config = config_manager or UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger("VaRCalculator")

        # Параметри з конфігурації
        risk_config = self.config.get('strategy.risk_management', {})
        self.confidence_levels = risk_config.get('var_confidence_levels', [0.95, 0.99])
        self.time_horizon = risk_config.get('var_time_horizon_days', 1)
        self.annual_trading_days = risk_config.get('annual_trading_days', 252)

    def calculate_var_historical(self,
                                returns: pd.Series,
                                confidence: float = 0.95,
                                time_horizon: int = 1) -> Dict[str, float]:
        """
        Historical Simulation VaR

        Args:
            returns: Історичні повернення
            confidence: Рівень довіри (0.95, 0.99)
            time_horizon: Часовий горизонт в днях

        Returns:
            Dict з VaR та допоміжними метриками
        """
        try:
            if len(returns) < 30:
                raise RiskManagementError("Недостатньо даних для VaR розрахунку")

            # Розрахунок percentile-based VaR
            var_pct = np.percentile(returns, (1 - confidence) * 100)

            # Масштабування на часовий горизонт
            if time_horizon > 1:
                # Припускаємо, що повернення нормально розподілені для scaling
                scaled_var = var_pct * np.sqrt(time_horizon)
            else:
                scaled_var = var_pct

            # Expected Shortfall (CVaR)
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
            self.logger.error(f"Помилка розрахунку Historical VaR: {e}")
            return {'error': str(e)}

    def calculate_var_parametric(self,
                                returns: pd.Series,
                                confidence: float = 0.95,
                                time_horizon: int = 1,
                                distribution: str = 'normal') -> Dict[str, float]:
        """
        Parametric VaR using normal or t-distribution

        Args:
            returns: Історичні повернення
            confidence: Рівень довіри
            time_horizon: Часовий горизонт
            distribution: 'normal' або 't'
        """
        try:
            if len(returns) < 30:
                raise RiskManagementError("Недостатньо даних")

            mu = returns.mean()
            sigma = returns.std()

            if distribution == 'normal':
                # Normal distribution VaR
                z_score = norm.ppf(1 - confidence)
                var = mu * time_horizon + sigma * np.sqrt(time_horizon) * z_score

                # CVaR for normal distribution
                alpha = 1 - confidence
                cvar = mu * time_horizon - sigma * np.sqrt(time_horizon) * norm.pdf(z_score) / alpha

            elif distribution == 't':
                # t-distribution fitting
                from scipy.stats import t as t_dist
                df, loc, scale = t_dist.fit(returns)
                t_score = t_dist.ppf(1 - confidence, df)
                var = loc * time_horizon + scale * np.sqrt(time_horizon) * t_score
                cvar = var  # Спрощена версія

            else:
                raise ValueError(f"Непідтримувана дистрибуція: {distribution}")

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
            self.logger.error(f"Помилка розрахунку Parametric VaR: {e}")
            return {'error': str(e)}

    def calculate_var_monte_carlo(self,
                                 returns: pd.Series,
                                 confidence: float = 0.95,
                                 time_horizon: int = 1,
                                 n_simulations: int = 10000) -> Dict[str, float]:
        """
        Monte Carlo VaR simulation

        Args:
            returns: Історичні повернення
            confidence: Рівень довіри
            time_horizon: Часовий горизонт
            n_simulations: Кількість симуляцій
        """
        try:
            if len(returns) < 30:
                raise RiskManagementError("Недостатньо даних")

            # Bootstrap sampling з заміною
            simulated_returns = []
            for _ in range(n_simulations):
                sample = np.random.choice(returns.values, size=time_horizon, replace=True)
                portfolio_return = np.prod(1 + sample) - 1
                simulated_returns.append(portfolio_return)

            simulated_returns = np.array(simulated_returns)
            var = np.percentile(simulated_returns, (1 - confidence) * 100)

            # CVaR
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
            self.logger.error(f"Помилка Monte Carlo VaR: {e}")
            return {'error': str(e)}

class StressTestingFramework:
    """
    Stress Testing Framework для сценаріїв кризи
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        self.config = config_manager or UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger("StressTesting")

        # Стандартні сценарії стресу
        self.scenarios = {
            'market_crash': {'shock': -0.15, 'description': '15% market crash'},
            'volatility_spike': {'volatility_multiplier': 3.0, 'description': '3x volatility increase'},
            'liquidity_crisis': {'liquidity_dryup': 0.8, 'description': '80% liquidity reduction'},
            'interest_rate_shock': {'rate_change': 0.025, 'description': '2.5% rate increase'},
            'correlated_crash': {'correlation_increase': 0.8, 'description': 'High correlation regime'}
        }

    def run_stress_test(self,
                       portfolio: Dict[str, float],
                       historical_data: pd.DataFrame,
                       scenario: str = 'market_crash') -> Dict[str, Any]:
        """
        Запуск stress test для портфеля

        Args:
            portfolio: Dict з {ticker: weight}
            historical_data: Історичні дані цін
            scenario: Назва сценарію

        Returns:
            Результати stress test
        """
        try:
            if scenario not in self.scenarios:
                raise ValueError(f"Невідомий сценарій: {scenario}")

            scenario_config = self.scenarios[scenario]

            # Розрахунок impact на портфель
            if scenario == 'market_crash':
                shock = scenario_config['shock']
                portfolio_impact = sum(weight * shock for weight in portfolio.values())

            elif scenario == 'volatility_spike':
                # Моделювання збільшення волатильності
                vol_multiplier = scenario_config['volatility_multiplier']
                # Спрощений розрахунок - збільшення VaR
                base_var = 0.02  # Припущення
                stressed_var = base_var * vol_multiplier
                portfolio_impact = -stressed_var * 2  # Conservative estimate

            elif scenario == 'liquidity_crisis':
                # Моделювання проблеми ліквідності
                liquidity_reduction = scenario_config['liquidity_dryup']
                portfolio_impact = -0.05 * liquidity_reduction  # 5% loss per 10% liquidity reduction

            else:
                portfolio_impact = -0.05  # Default 5% loss

            # Розрахунок recovery time (спрощено)
            recovery_days = abs(portfolio_impact) * 100  # Rough estimate

            return {
                'scenario': scenario,
                'description': scenario_config['description'],
                'portfolio_impact': float(portfolio_impact),
                'portfolio_loss_pct': float(abs(portfolio_impact) * 100),
                'estimated_recovery_days': int(recovery_days),
                'breaches_limits': abs(portfolio_impact) > 0.1,  # 10% loss threshold
                'recommendations': self._generate_recommendations(portfolio_impact, scenario)
            }

        except Exception as e:
            self.logger.error(f"Помилка stress test: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self, impact: float, scenario: str) -> List[str]:
        """Генерація рекомендацій на основі результатів"""
        recommendations = []

        if abs(impact) > 0.1:  # >10% loss
            recommendations.append("Критично: розглянути часткове закриття позицій")
            recommendations.append("Рекомендується збільшити stop-loss рівні")

        if scenario == 'market_crash':
            recommendations.append("Розглянути хеджування через опціони чи інверсні ETF")
            recommendations.append("Переглянути exposure до ризикових активів")

        elif scenario == 'volatility_spike':
            recommendations.append("Зменшити розмір позицій для зниження волатильності")
            recommendations.append("Розглянути hedging стратегії")

        return recommendations

class LiquidityRiskAssessor:
    """
    Оцінка ризику ліквідності
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        self.config = config_manager or UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger("LiquidityRisk")

    def assess_liquidity_risk(self,
                            ticker: str,
                            volume_data: pd.Series,
                            price_data: pd.Series,
                            position_size: float) -> Dict[str, Any]:
        """
        Оцінка ліквідності активу

        Args:
            ticker: Тікер активу
            volume_data: Дані обсягів торгів
            price_data: Дані цін
            position_size: Розмір позиції в доларах

        Returns:
            Метрики ліквідності
        """
        try:
            # Average daily volume
            avg_daily_volume = volume_data.mean()
            avg_daily_volume_dollars = (volume_data * price_data).mean()

            # Bid-ask spread proxy (спрощено)
            returns = price_data.pct_change()
            volatility = returns.std()
            spread_estimate = volatility * 0.01  # Rough estimate

            # Market impact estimation
            market_impact_pct = min(position_size / avg_daily_volume_dollars, 0.1)  # Max 10%

            # Liquidity score (0-100, higher = more liquid)
            volume_score = min(avg_daily_volume / 1000000, 1.0)  # Normalize to $1M
            spread_score = max(0, 1 - spread_estimate * 100)  # Lower spread = higher score
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3) * 100

            # Risk assessment
            if liquidity_score < 30:
                risk_level = "HIGH"
                risk_description = "Високий ризик ліквідності - уникати великих позицій"
            elif liquidity_score < 60:
                risk_level = "MEDIUM"
                risk_description = "Середній ризик ліквідності - обмежити розмір позицій"
            else:
                risk_level = "LOW"
                risk_description = "Низький ризик ліквідності - прийнятно для торгів"

            return {
                'ticker': ticker,
                'liquidity_score': float(liquidity_score),
                'risk_level': risk_level,
                'risk_description': risk_description,
                'avg_daily_volume': float(avg_daily_volume),
                'avg_daily_volume_dollars': float(avg_daily_volume_dollars),
                'estimated_spread_pct': float(spread_estimate),
                'market_impact_pct': float(market_impact_pct),
                'max_position_size': float(avg_daily_volume_dollars * 0.01),  # 1% of daily volume
                'recommendations': self._generate_liquidity_recommendations(risk_level, position_size, avg_daily_volume_dollars)
            }

        except Exception as e:
            self.logger.error(f"Помилка оцінки ліквідності: {e}")
            return {'error': str(e)}

    def _generate_liquidity_recommendations(self,
                                          risk_level: str,
                                          position_size: float,
                                          avg_daily_volume: float) -> List[str]:
        """Генерація рекомендацій по ліквідності"""
        recommendations = []

        if risk_level == "HIGH":
            recommendations.append("Уникати цього активу або використовувати дуже малі позиції")
            recommendations.append("Розглянути альтернативні активи з кращою ліквідністю")

        elif risk_level == "MEDIUM":
            max_safe_size = avg_daily_volume * 0.005  # 0.5% of daily volume
            if position_size > max_safe_size:
                recommendations.append(f"Зменшити позицію нижче ${max_safe_size:,.0f} для безпечної торгівлі")

        return recommendations

class RiskLimitsManager:
    """
    Управління лімітам ризику
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        self.config = config_manager or UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger("RiskLimits")

        # Завантаження лімітів з конфігурації
        risk_config = self.config.get('strategy.risk_management', {})
        self.limits = {
            'max_portfolio_var': risk_config.get('max_portfolio_var_pct', 0.05),  # 5%
            'max_single_position': risk_config.get('max_single_position_pct', 0.10),  # 10%
            'max_daily_loss': risk_config.get('max_daily_loss_pct', 0.03),  # 3%
            'max_drawdown': risk_config.get('max_drawdown_pct', 0.15),  # 15%
            'max_leverage': risk_config.get('max_leverage', 2.0),  # 2x
        }

    def check_limits(self,
                    portfolio_value: float,
                    positions: Dict[str, Dict[str, Any]],
                    daily_pnl: float,
                    current_drawdown: float) -> Dict[str, Any]:
        """
        Перевірка дотримання лімітів ризику

        Args:
            portfolio_value: Вартість портфеля
            positions: Dict з позиціями {ticker: {'size': float, 'value': float}}
            daily_pnl: Денний P&L
            current_drawdown: Поточний drawdown

        Returns:
            Результат перевірки лімітів
        """
        violations = []
        warnings = []

        # Перевірка VaR ліміту (спрощено - використовуємо фіксований VaR)
        estimated_var = portfolio_value * 0.02  # 2% VaR assumption
        if estimated_var > portfolio_value * self.limits['max_portfolio_var']:
            violations.append({
                'type': 'portfolio_var',
                'current': estimated_var / portfolio_value,
                'limit': self.limits['max_portfolio_var'],
                'message': f"Portfolio VaR {estimated_var/portfolio_value:.1%} перевищує ліміт {self.limits['max_portfolio_var']:.1%}"
            })

        # Перевірка позицій
        for ticker, pos_data in positions.items():
            position_pct = pos_data['value'] / portfolio_value
            if position_pct > self.limits['max_single_position']:
                violations.append({
                    'type': 'single_position',
                    'ticker': ticker,
                    'current': position_pct,
                    'limit': self.limits['max_single_position'],
                    'message': f"Позиція {ticker} {position_pct:.1%} перевищує ліміт {self.limits['max_single_position']:.1%}"
                })

        # Перевірка денних втрат
        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        if daily_loss_pct > self.limits['max_daily_loss']:
            violations.append({
                'type': 'daily_loss',
                'current': daily_loss_pct,
                'limit': self.limits['max_daily_loss'],
                'message': f"Денні втрати {daily_loss_pct:.1%} перевищують ліміт {self.limits['max_daily_loss']:.1%}"
            })

        # Перевірка drawdown
        if current_drawdown > self.limits['max_drawdown']:
            violations.append({
                'type': 'drawdown',
                'current': current_drawdown,
                'limit': self.limits['max_drawdown'],
                'message': f"Drawdown {current_drawdown:.1%} перевищує ліміт {self.limits['max_drawdown']:.1%}"
            })

        # Warnings для наближення до лімітів
        if estimated_var / portfolio_value > self.limits['max_portfolio_var'] * 0.8:
            warnings.append("Portfolio VaR наближається до ліміту")

        if daily_loss_pct > self.limits['max_daily_loss'] * 0.7:
            warnings.append("Денні втрати наближаються до ліміту")

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
    Головний клас Risk Management Framework
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        self.config = config_manager or UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger("RiskFramework")

        # Ініціалізація компонентів
        self.var_calculator = VaRCalculator(self.config)
        self.stress_tester = StressTestingFramework(self.config)
        self.liquidity_assessor = LiquidityRiskAssessor(self.config)
        self.limits_manager = RiskLimitsManager(self.config)

        self.logger.info("Risk Management Framework ініціалізовано")

    def comprehensive_risk_assessment(self,
                                    portfolio: Dict[str, float],
                                    historical_data: pd.DataFrame,
                                    current_positions: Dict[str, Dict[str, Any]],
                                    portfolio_value: float,
                                    daily_pnl: float = 0.0,
                                    current_drawdown: float = 0.0) -> Dict[str, Any]:
        """
        Комплексна оцінка ризиків портфеля

        Args:
            portfolio: Dict з вагами {ticker: weight}
            historical_data: Історичні дані
            current_positions: Поточні позиції
            portfolio_value: Вартість портфеля
            daily_pnl: Денний P&L
            current_drawdown: Поточний drawdown

        Returns:
            Повний звіт по ризиках
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

            # 1. VaR розрахунки для кожного активу та портфеля
            portfolio_returns = self._calculate_portfolio_returns(portfolio, historical_data)

            if len(portfolio_returns) > 0:
                report['risk_metrics']['portfolio_var'] = self.var_calculator.calculate_var_historical(
                    portfolio_returns, confidence=0.95
                )
                report['risk_metrics']['portfolio_var_99'] = self.var_calculator.calculate_var_historical(
                    portfolio_returns, confidence=0.99
                )

            # 2. Stress testing
            for scenario in ['market_crash', 'volatility_spike', 'liquidity_crisis']:
                report['stress_tests'][scenario] = self.stress_tester.run_stress_test(
                    portfolio, historical_data, scenario
                )

            # 3. Liquidity analysis
            report['liquidity_analysis'] = {}
            for ticker in portfolio.keys():
                if ticker in historical_data.columns:
                    ticker_data = historical_data[ticker]
                    position_value = portfolio_value * portfolio[ticker]
                    report['liquidity_analysis'][ticker] = self.liquidity_assessor.assess_liquidity_risk(
                        ticker, ticker_data, ticker_data, position_value
                    )

            # 4. Limits check
            report['limits_check'] = self.limits_manager.check_limits(
                portfolio_value, current_positions, daily_pnl, current_drawdown
            )

            # 5. Генерація рекомендацій
            report['recommendations'] = self._generate_comprehensive_recommendations(report)
            report['alerts'] = self._generate_alerts(report)

            self.logger.info("Комплексна оцінка ризиків завершена")
            return report

        except Exception as e:
            self.logger.error(f"Помилка комплексної оцінки ризиків: {e}")
            return {'error': str(e)}

    def _calculate_portfolio_returns(self, portfolio: Dict[str, float], data: pd.DataFrame) -> pd.Series:
        """Розрахунок повернень портфеля"""
        try:
            portfolio_returns = pd.Series(0.0, index=data.index)

            for ticker, weight in portfolio.items():
                if ticker in data.columns:
                    asset_returns = data[ticker].pct_change()
                    portfolio_returns += asset_returns * weight

            return portfolio_returns.dropna()

        except Exception as e:
            self.logger.error(f"Помилка розрахунку повернень портфеля: {e}")
            return pd.Series()

    def _generate_comprehensive_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Генерація комплексних рекомендацій"""
        recommendations = []

        # Перевірка VaR
        if 'portfolio_var' in report['risk_metrics']:
            var_95 = report['risk_metrics']['portfolio_var'].get('var', 0)
            if var_95 < -0.05:  # >5% potential loss
                recommendations.append("Високий VaR: розглянути зменшення ризику портфеля")

        # Перевірка stress tests
        for scenario, result in report['stress_tests'].items():
            if result.get('breaches_limits', False):
                recommendations.append(f"Stress test '{scenario}': переглянути стратегію ризик-менеджменту")

        # Перевірка ліквідності
        for ticker, analysis in report['liquidity_analysis'].items():
            if analysis.get('risk_level') == 'HIGH':
                recommendations.append(f"Високий ризик ліквідності для {ticker}: зменшити позицію")

        # Перевірка лімітів
        if not report['limits_check'].get('limits_respected', True):
            recommendations.append("Порушення лімітів ризику: негайно скоригувати позиції")

        return recommendations

    def _generate_alerts(self, report: Dict[str, Any]) -> List[str]:
        """Генерація алертів"""
        alerts = []

        # Критичні порушення
        if not report['limits_check'].get('limits_respected', True):
            alerts.append("КРИТИЧНО: Порушення лімітів ризику!")

        # Високий VaR
        if 'portfolio_var' in report['risk_metrics']:
            var_99 = report['risk_metrics']['portfolio_var_99'].get('var', 0)
            if var_99 < -0.10:  # >10% potential loss at 99% confidence
                alerts.append("КРИТИЧНО: Екстремальний VaR на 99% рівні довіри!")

        return alerts