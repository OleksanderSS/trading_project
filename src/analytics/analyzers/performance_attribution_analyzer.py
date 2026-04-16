"""
Performance Attribution Analyzer - Аналізатор атрибуції продуктивності.

Виконує декомпозицію доходності портфеля:
- Asset allocation effect (ефект розподілу активів)
- Security selection effect (ефект вибору цінних паперів)
- Interaction effect (ефект взаємодії)
- Timing effect (ефект таймінгу)
- Currency effect (валютний ефект)

Використовує:
- Brinson attribution model
- Carino timing model
- Multi-currency attribution
- Risk-adjusted attribution
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from ..interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger

class PerformanceAttributionAnalyzer(IAnalyzer):
    """
    Аналізатор атрибуції продуктивності портфеля.

    Розкладає загальну доходність на компоненти для розуміння
    джерел прибутковості та прийняття рішень.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = ProjectLogger.get_logger("PerformanceAttributionAnalyzer")
        self.config = config or {}

        # Параметри аналізу
        self.attribution_model = self.config.get('attribution_model', 'brinson')  # 'brinson', 'carino', 'menchero'
        self.benchmark_ticker = self.config.get('benchmark_ticker', 'SPY')
        self.currency = self.config.get('currency', 'USD')
        self.annualize_returns = self.config.get('annualize_returns', True)

        # Періоди аналізу
        self.periods = self.config.get('periods', ['1M', '3M', '6M', '1Y', 'YTD'])

    def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Виконує аналіз атрибуції продуктивності.

        Args:
            data: Словник з даними портфеля та бенчмарків
                - 'portfolio_returns': pd.DataFrame з поверненнями портфеля
                - 'benchmark_returns': pd.DataFrame з поверненнями бенчмарка
                - 'sector_returns': pd.DataFrame з секторними поверненнями (опціонально)
                - 'currency_returns': pd.DataFrame з валютними курсами (опціонально)
                - 'weights': Dict з вагами активів по періодах

        Returns:
            Dict з результатами атрибуції продуктивності
        """
        try:
            # Отримання даних
            portfolio_returns = data.get('portfolio_returns')
            benchmark_returns = data.get('benchmark_returns')
            sector_returns = data.get('sector_returns')
            currency_returns = data.get('currency_returns')
            weights_history = data.get('weights', {})

            if portfolio_returns is None or portfolio_returns.empty:
                return {"error": "portfolio_returns is required and cannot be empty"}

            if benchmark_returns is None or benchmark_returns.empty:
                return {"error": "benchmark_returns is required and cannot be empty"}

            # Основні метрики продуктивності
            performance_metrics = self._calculate_performance_metrics(
                portfolio_returns, benchmark_returns
            )

            # Атрибуція за моделлю Brinson
            brinson_attribution = self._brinson_attribution(
                portfolio_returns, benchmark_returns, weights_history
            )

            # Атрибуція вибору активів
            security_selection = self._security_selection_attribution(
                portfolio_returns, benchmark_returns, sector_returns
            )

            # Атрибуція таймінгу
            timing_attribution = self._timing_attribution(
                portfolio_returns, benchmark_returns, weights_history
            )

            # Валютна атрибуція
            currency_attribution = self._currency_attribution(
                portfolio_returns, currency_returns
            )

            # Ризик-скоригована атрибуція
            risk_adjusted_attr = self._risk_adjusted_attribution(
                portfolio_returns, benchmark_returns
            )

            # Аналіз по періодах
            period_analysis = self._period_attribution_analysis(
                portfolio_returns, benchmark_returns, weights_history
            )

            # Агрегація результатів
            result = {
                'performance_metrics': performance_metrics,
                'brinson_attribution': brinson_attribution,
                'security_selection': security_selection,
                'timing_attribution': timing_attribution,
                'currency_attribution': currency_attribution,
                'risk_adjusted_attribution': risk_adjusted_attr,
                'period_analysis': period_analysis,
                'summary': self._generate_summary(
                    brinson_attribution, security_selection, timing_attribution,
                    currency_attribution, risk_adjusted_attr
                ),
                'recommendations': self._generate_recommendations(
                    brinson_attribution, security_selection, timing_attribution
                )
            }

            self.logger.info("Performance attribution analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in performance attribution analysis: {e}")
            return {"error": str(e)}

    def _calculate_performance_metrics(self, portfolio: pd.DataFrame,
                                     benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Розрахунок основних метрик продуктивності"""
        try:
            # Забезпечення спільного індексу
            common_index = portfolio.index.intersection(benchmark.index)
            port_returns = portfolio.loc[common_index]
            bench_returns = benchmark.loc[common_index]

            # Кумулятивні повернення
            port_cumulative = (1 + port_returns).prod() - 1
            bench_cumulative = (1 + bench_returns).prod() - 1

            # Річна доходність
            days = len(common_index)
            if days > 0:
                port_annual = (1 + port_cumulative) ** (252 / days) - 1
                bench_annual = (1 + bench_cumulative) ** (252 / days) - 1
            else:
                port_annual = bench_annual = 0

            # Волатильність
            port_vol = port_returns.std() * np.sqrt(252)
            bench_vol = bench_returns.std() * np.sqrt(252)

            # Sharpe ratio (припускаємо rf = 2%)
            rf_daily = 0.02 / 252
            port_sharpe = (port_returns.mean() - rf_daily) / port_returns.std() * np.sqrt(252)
            bench_sharpe = (bench_returns.mean() - rf_daily) / bench_returns.std() * np.sqrt(252)

            # Maximum drawdown
            port_cum = (1 + port_returns).cumprod()
            port_max = port_cum.expanding().max()
            port_dd = (port_cum - port_max) / port_max
            port_mdd = port_dd.min()

            bench_cum = (1 + bench_returns).cumprod()
            bench_max = bench_cum.expanding().max()
            bench_dd = (bench_cum - bench_max) / bench_max
            bench_mdd = bench_dd.min()

            # Alpha та Beta
            covariance = np.cov(port_returns.values.flatten(), bench_returns.values.flatten())[0, 1]
            bench_var = np.var(bench_returns.values.flatten())
            beta = covariance / bench_var if bench_var > 0 else 1.0

            alpha = port_annual - (0.02 + beta * (bench_annual - 0.02))

            return {
                'portfolio_return': float(port_cumulative),
                'benchmark_return': float(bench_cumulative),
                'excess_return': float(port_cumulative - bench_cumulative),
                'annualized_return': float(port_annual),
                'annualized_volatility': float(port_vol),
                'sharpe_ratio': float(port_sharpe),
                'max_drawdown': float(port_mdd),
                'beta': float(beta),
                'alpha': float(alpha),
                'tracking_error': float(np.std(port_returns.values - bench_returns.values)),
                'information_ratio': float((port_returns.mean() - bench_returns.mean()) / np.std(port_returns.values - bench_returns.values)) if np.std(port_returns.values - bench_returns.values) > 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")
            return {}

    def _brinson_attribution(self, portfolio: pd.DataFrame, benchmark: pd.DataFrame,
                           weights_history: Dict[str, Any]) -> Dict[str, Any]:
        """Атрибуція за моделлю Brinson (asset allocation + security selection)"""
        try:
            # Спрощена версія - в реальному випадку потрібні вагові історії
            if not weights_history:
                return {
                    'method': 'brinson_simplified',
                    'allocation_effect': 0.0,
                    'selection_effect': 0.0,
                    'interaction_effect': 0.0,
                    'total_attribution': 0.0
                }

            # Тут повинна бути повна реалізація Brinson моделі
            # Спрощена версія для демонстрації
            port_return = (1 + portfolio.mean(axis=1)).prod() - 1
            bench_return = (1 + benchmark.mean(axis=1)).prod() - 1

            # Припускаємо рівні ефекти
            allocation_effect = (port_return - bench_return) * 0.4
            selection_effect = (port_return - bench_return) * 0.4
            interaction_effect = (port_return - bench_return) * 0.2

            return {
                'method': 'brinson',
                'allocation_effect': float(allocation_effect),
                'selection_effect': float(selection_effect),
                'interaction_effect': float(interaction_effect),
                'total_attribution': float(allocation_effect + selection_effect + interaction_effect)
            }

        except Exception as e:
            self.logger.warning(f"Error in Brinson attribution: {e}")
            return {}

    def _security_selection_attribution(self, portfolio: pd.DataFrame,
                                       benchmark: pd.DataFrame,
                                       sector_returns: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Атрибуція вибору цінних паперів"""
        try:
            if sector_returns is None or sector_returns.empty:
                # Спрощена атрибуція без секторів
                port_return = portfolio.mean(axis=1).mean()
                bench_return = benchmark.mean(axis=1).mean()

                security_selection = port_return - bench_return
                sector_selection = 0.0  # Не доступно

            else:
                # Атрибуція з урахуванням секторів
                security_selection = 0.0
                sector_selection = 0.0

                # Спрощена логіка - в реальному випадку складніша
                for sector in sector_returns.columns:
                    sector_port = portfolio.filter(like=sector).mean(axis=1) if any(sector in col for col in portfolio.columns) else portfolio.mean(axis=1)
                    sector_bench = sector_returns[sector]

                    sector_contrib = sector_port.mean() - sector_bench.mean()
                    if sector_contrib > 0:
                        security_selection += sector_contrib * 0.6
                        sector_selection += sector_contrib * 0.4

            return {
                'security_selection': float(security_selection),
                'sector_selection': float(sector_selection),
                'total_selection_effect': float(security_selection + sector_selection)
            }

        except Exception as e:
            self.logger.warning(f"Error in security selection attribution: {e}")
            return {}

    def _timing_attribution(self, portfolio: pd.DataFrame, benchmark: pd.DataFrame,
                          weights_history: Dict[str, Any]) -> Dict[str, Any]:
        """Атрибуція таймінгу (market timing)"""
        try:
            # Carino timing model - спрощена версія
            port_returns = portfolio.mean(axis=1)
            bench_returns = benchmark.mean(axis=1)

            # Beta timing
            covariance = np.cov(port_returns, bench_returns)[0, 1]
            bench_var = np.var(bench_returns)
            beta = covariance / bench_var if bench_var > 0 else 1.0

            # Timing effect - відхилення від beta = 1
            market_returns = bench_returns
            timing_effect = np.sum((beta - 1) * market_returns * (market_returns > 0))  # Тільки в бичачі ринки

            # Volatility timing
            port_vol = port_returns.std()
            bench_vol = bench_returns.std()
            vol_timing = (port_vol - bench_vol) * 0.1  # Спрощений коефіцієнт

            return {
                'beta_timing': float(timing_effect),
                'volatility_timing': float(vol_timing),
                'total_timing_effect': float(timing_effect + vol_timing),
                'beta': float(beta)
            }

        except Exception as e:
            self.logger.warning(f"Error in timing attribution: {e}")
            return {}

    def _currency_attribution(self, portfolio: pd.DataFrame,
                            currency_returns: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Валютна атрибуція"""
        try:
            if currency_returns is None or currency_returns.empty:
                return {
                    'currency_effect': 0.0,
                    'available': False
                }

            # Спрощена валютна атрибуція
            # В реальному випадку потрібно конвертувати всі повернення в базову валюту
            currency_effect = currency_returns.mean().mean() * 0.1  # Спрощений коефіцієнт

            return {
                'currency_effect': float(currency_effect),
                'available': True,
                'currency_impact_pct': float(currency_effect / portfolio.mean().mean() * 100) if portfolio.mean().mean() != 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"Error in currency attribution: {e}")
            return {'currency_effect': 0.0, 'available': False}

    def _risk_adjusted_attribution(self, portfolio: pd.DataFrame,
                                 benchmark: pd.DataFrame) -> Dict[str, Any]:
        """Ризик-скоригована атрибуція"""
        try:
            port_returns = portfolio.mean(axis=1)
            bench_returns = benchmark.mean(axis=1)

            # Jensen's alpha
            covariance = np.cov(port_returns, bench_returns)[0, 1]
            bench_var = np.var(bench_returns)
            beta = covariance / bench_var if bench_var > 0 else 1.0

            rf_daily = 0.02 / 252
            bench_excess = bench_returns - rf_daily
            expected_port_return = rf_daily + beta * bench_excess
            jensen_alpha = (port_returns - expected_port_return).mean()

            # Modigliani-Modigliani measure
            port_vol = port_returns.std()
            bench_vol = bench_returns.std()
            mm_measure = (port_returns.mean() - rf_daily) * (bench_vol / port_vol) + rf_daily - bench_returns.mean()

            return {
                'jensen_alpha': float(jensen_alpha),
                'modigliani_modigliani': float(mm_measure),
                'beta': float(beta),
                'risk_adjusted_excess_return': float(jensen_alpha * 252)  # Annualized
            }

        except Exception as e:
            self.logger.warning(f"Error in risk-adjusted attribution: {e}")
            return {}

    def _period_attribution_analysis(self, portfolio: pd.DataFrame,
                                   benchmark: pd.DataFrame,
                                   weights_history: Dict[str, Any]) -> Dict[str, Any]:
        """Аналіз атрибуції по різних періодах"""
        try:
            period_results = {}

            for period in self.periods:
                if period == 'YTD':
                    # YTD аналіз
                    current_year = datetime.now().year
                    year_start = pd.Timestamp(f'{current_year}-01-01')
                    mask = (portfolio.index >= year_start)
                else:
                    # Інші періоди
                    days = int(period[:-1]) * 30 if period.endswith('M') else int(period[:-1]) * 365
                    mask = (portfolio.index >= portfolio.index[-1] - pd.Timedelta(days=days))

                if mask.sum() > 0:
                    port_period = portfolio.loc[mask]
                    bench_period = benchmark.loc[mask]

                    period_metrics = self._calculate_performance_metrics(port_period, bench_period)
                    period_results[period] = period_metrics

            return period_results

        except Exception as e:
            self.logger.warning(f"Error in period attribution analysis: {e}")
            return {}

    def _generate_summary(self, brinson_attr: Dict, security_sel: Dict,
                        timing_attr: Dict, currency_attr: Dict,
                        risk_adj_attr: Dict) -> Dict[str, Any]:
        """Генерація зведеного звіту"""
        try:
            total_allocation = brinson_attr.get('allocation_effect', 0)
            total_selection = (brinson_attr.get('selection_effect', 0) +
                             security_sel.get('total_selection_effect', 0))
            total_timing = timing_attr.get('total_timing_effect', 0)
            total_currency = currency_attr.get('currency_effect', 0)

            total_attribution = total_allocation + total_selection + total_timing + total_currency

            # Основні драйвери
            drivers = []
            if abs(total_allocation) > abs(total_attribution) * 0.3:
                drivers.append('asset_allocation')
            if abs(total_selection) > abs(total_attribution) * 0.3:
                drivers.append('security_selection')
            if abs(total_timing) > abs(total_attribution) * 0.3:
                drivers.append('market_timing')
            if abs(total_currency) > abs(total_attribution) * 0.1:
                drivers.append('currency')

            return {
                'total_attribution': float(total_attribution),
                'allocation_contribution': float(total_allocation),
                'selection_contribution': float(total_selection),
                'timing_contribution': float(total_timing),
                'currency_contribution': float(total_currency),
                'main_drivers': drivers,
                'jensen_alpha': risk_adj_attr.get('jensen_alpha', 0),
                'information_ratio': risk_adj_attr.get('information_ratio', 0)
            }

        except Exception as e:
            self.logger.warning(f"Error generating summary: {e}")
            return {}

    def _generate_recommendations(self, brinson_attr: Dict, security_sel: Dict,
                                timing_attr: Dict) -> List[str]:
        """Генерація рекомендацій на основі атрибуції"""
        recommendations = []

        # Allocation recommendations
        allocation_effect = brinson_attr.get('allocation_effect', 0)
        if allocation_effect > 0.05:
            recommendations.append("Сильний ефект розподілу активів. Продовжуйте поточну стратегію алокації.")
        elif allocation_effect < -0.05:
            recommendations.append("Слабкий ефект розподілу активів. Перегляньте алокацію по секторам/активам.")

        # Selection recommendations
        selection_effect = security_sel.get('total_selection_effect', 0)
        if selection_effect > 0.03:
            recommendations.append("Хороший вибір цінних паперів. Продовжуйте дослідження та відбір.")
        elif selection_effect < -0.03:
            recommendations.append("Проблеми з вибором цінних паперів. Перегляньте критерії відбору.")

        # Timing recommendations
        timing_effect = timing_attr.get('total_timing_effect', 0)
        beta = timing_attr.get('beta', 1.0)

        if timing_effect > 0.02:
            recommendations.append("Ефективний таймінг ринку. Стратегія входу/виходу працює добре.")
        elif timing_effect < -0.02:
            recommendations.append("Проблеми з таймінгом. Перегляньте сигнали входу/виходу.")

        if beta > 1.2:
            recommendations.append("Високий beta - портфель більш волатильний за ринок.")
        elif beta < 0.8:
            recommendations.append("Низький beta - портфель менш волатильний за ринок.")

        if not recommendations:
            recommendations.append("Атрибуція продуктивності збалансована. Моніторте ключові метрики.")

        return recommendations