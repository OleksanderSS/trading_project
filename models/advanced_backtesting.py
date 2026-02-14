#!/usr/bin/env python3
"""
Advanced Backtesting Module
Розширений бектестинг: walk-forward, transaction costs, market impact, slippage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

from models.fama_french_factors import get_fama_french_factors
from models.hedge_fund_analyzer import calculate_performance_metrics

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """
    Розширений бектестер з реалістичними умовами
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AdvancedBacktester")
        
        # Параметри транзакційних витрат
        self.transaction_costs = {
            'commission': 0.001,      # 0.1% комісія
            'spread': 0.0005,         # 0.05% спред
            'slippage_rate': 0.0001,  # 0.01% slippage
            'financing_rate': 0.0002, # 0.02% фінансування
            'stamp_duty': 0.0001      # 0.01% податок
        }
        
        # Параметри market impact
        self.market_impact = {
            'linear_coefficient': 0.00001,   # Лінійний коефіцієнт
            'square_root_coefficient': 0.0001, # sqrt коефіцієнт
            'daily_volume_factor': 0.1,      # Фактор обсягу
            'volatility_factor': 0.5         # Фактор волатильності
        }
        
        # Параметри walk-forward
        self.walk_forward_params = {
            'train_period': 252,      # 1 рік тренування
            'test_period': 63,        # 3 місяці тестування
            'step_size': 21,          # 1 місяць крок
            'min_train_periods': 126  # Мінімум 6 місяців
        }
        
        # Ризик-менеджмент
        self.risk_params = {
            'max_position_size': 0.2,    # 20% макс позиція
            'max_portfolio_risk': 0.15,  # 15% макс ризик портфоліо
            'stop_loss': 0.05,           # 5% stop loss
            'take_profit': 0.10,         # 10% take profit
            'max_leverage': 2.0           # Макс кредитне плече
        }
        
        self.logger.info("AdvancedBacktester initialized")
    
    def calculate_transaction_costs(self, trade_size: float, price: float, 
                                  volume: float = None, volatility: float = None) -> Dict[str, float]:
        """
        Розрахувати повні транзакційні витрати
        
        Args:
            trade_size: Розмір торгівлі (в грошовому вираженні)
            price: Ціна активу
            volume: Денний обсяг (опціонально)
            volatility: Волатильність (опціонально)
            
        Returns:
            Dict: Деталізація витрат
        """
        try:
            costs = {}
            
            # Базові витрати
            costs['commission'] = trade_size * self.transaction_costs['commission']
            costs['spread'] = trade_size * self.transaction_costs['spread']
            costs['stamp_duty'] = trade_size * self.transaction_costs['stamp_duty']
            
            # Slippage (залежить від обсягу та волатильності)
            if volume is not None and volatility is not None:
                # Market impact model
                volume_ratio = abs(trade_size) / (volume * price)
                
                # Linear impact
                linear_impact = self.market_impact['linear_coefficient'] * volume_ratio
                
                # Square root impact
                sqrt_impact = self.market_impact['square_root_coefficient'] * np.sqrt(volume_ratio)
                
                # Volatility adjustment
                vol_adjustment = 1 + volatility * self.market_impact['volatility_factor']
                
                # Total impact
                market_impact = (linear_impact + sqrt_impact) * vol_adjustment
                costs['market_impact'] = abs(trade_size) * market_impact
                
                # Slippage
                costs['slippage'] = abs(trade_size) * self.transaction_costs['slippage_rate'] * vol_adjustment
            else:
                # Базовий slippage
                costs['slippage'] = abs(trade_size) * self.transaction_costs['slippage_rate']
                costs['market_impact'] = 0
            
            # Фінансування (для коротких позицій)
            if trade_size < 0:
                costs['financing'] = abs(trade_size) * self.transaction_costs['financing_rate']
            else:
                costs['financing'] = 0
            
            # Загальні витрати
            costs['total'] = sum(costs.values())
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction costs: {e}")
            return {'total': 0}
    
    def simulate_trade_execution(self, signal: float, price: float, 
                              volume: float, volatility: float,
                              current_position: float = 0) -> Dict[str, any]:
        """
        Симуляція виконання торгівлі з реалістичними умовами
        
        Args:
            signal: Сигнал (від -1 до 1)
            price: Поточна ціна
            volume: Денний обсяг
            volatility: Волатильність
            current_position: Поточна позиція
            
        Returns:
            Dict: Результати торгівлі
        """
        try:
            # Розраховуємо бажаний розмір позиції
            max_trade_size = price * volume * 0.1  # 10% від деного обсягу
            desired_position = signal * max_trade_size
            
            # Розраховуємо зміну позиції
            trade_size = desired_position - current_position
            
            # Обмежуємо розмір торгівлі
            max_position = self.risk_params['max_position_size'] * 1000000  # Припустимо $1M портфель
            trade_size = np.clip(trade_size, -max_position, max_position)
            
            if abs(trade_size) < 1000:  # Мінімальна торгівля $1000
                return {'executed': False, 'reason': 'Trade too small'}
            
            # Розраховуємо транзакційні витрати
            costs = self.calculate_transaction_costs(trade_size, price, volume, volatility)
            
            # Розраховуємо ціну виконання (з урахуванням slippage)
            if trade_size > 0:  # Купівля
                execution_price = price * (1 + costs['slippage'] / abs(trade_size))
            else:  # Продаж
                execution_price = price * (1 - costs['slippage'] / abs(trade_size))
            
            # Результати торгівлі
            result = {
                'executed': True,
                'signal': signal,
                'trade_size': trade_size,
                'execution_price': execution_price,
                'transaction_costs': costs,
                'new_position': current_position + trade_size,
                'cost_bps': costs['total'] / abs(trade_size) * 10000  # в базисних пунктах
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error simulating trade execution: {e}")
            return {'executed': False, 'error': str(e)}
    
    def walk_forward_analysis(self, data: pd.DataFrame, signals: pd.DataFrame,
                            train_model_func, predict_func,
                            train_params: Dict = None) -> Dict[str, any]:
        """
        Walk-forward аналіз
        
        Args:
            data: Дані для тренування/тестування
            signals: Сигнали
            train_model_func: Функція тренування моделі
            predict_func: Функція прогнозування
            train_params: Параметри тренування
            
        Returns:
            Dict: Результати walk-forward аналізу
        """
        try:
            if train_params is None:
                train_params = {}
            
            # Параметри walk-forward
            train_period = self.walk_forward_params['train_period']
            test_period = self.walk_forward_params['test_period']
            step_size = self.walk_forward_params['step_size']
            
            results = {
                'periods': [],
                'overall_performance': {},
                'model_performance': [],
                'transaction_costs': [],
                'risk_metrics': []
            }
            
            # Ітеруємо по періодах
            start_idx = 0
            period_count = 0
            
            while start_idx + train_period + test_period <= len(data):
                period_count += 1
                
                # Визначаємо індекси
                train_start = start_idx
                train_end = start_idx + train_period
                test_start = train_end
                test_end = train_end + test_period
                
                # Дані для тренування
                train_data = data.iloc[train_start:train_end]
                train_signals = signals.iloc[train_start:train_end]
                
                # Дані для тестування
                test_data = data.iloc[test_start:test_end]
                test_signals = signals.iloc[test_start:test_end]
                
                # Тренуємо модель
                model = train_model_func(train_data, train_signals, **train_params)
                
                # Робимо прогнози
                predictions = predict_func(model, test_data)
                
                # Бектест на тестовому періоді
                period_results = self._backtest_period(
                    test_data, predictions, test_signals
                )
                
                # Зберігаємо результати періоду
                period_results['period'] = period_count
                period_results['train_start'] = data.index[train_start]
                period_results['train_end'] = data.index[train_end-1]
                period_results['test_start'] = data.index[test_start]
                period_results['test_end'] = data.index[test_end-1]
                
                results['periods'].append(period_results)
                results['model_performance'].append(period_results.get('performance', {}))
                results['transaction_costs'].append(period_results.get('total_costs', 0))
                
                # Наступний період
                start_idx += step_size
                
                self.logger.info(f"Walk-forward period {period_count} completed")
            
            # Розраховуємо загальні результати
            if results['periods']:
                results['overall_performance'] = self._calculate_overall_performance(results['periods'])
                results['average_transaction_costs'] = np.mean(results['transaction_costs'])
                results['success_rate'] = sum(1 for p in results['periods'] 
                                           if p.get('performance', {}).get('sharpe_ratio', 0) > 0) / len(results['periods'])
            
            self.logger.info(f"Walk-forward analysis completed: {period_count} periods")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {'error': str(e)}
    
    def _backtest_period(self, data: pd.DataFrame, predictions: pd.Series,
                        signals: pd.DataFrame) -> Dict[str, any]:
        """
        Бектест для одного періоду
        """
        try:
            # Ініціалізація
            portfolio_value = 1000000  # $1M стартовий капітал
            positions = {}
            trades = []
            daily_returns = []
            
            # Ітеруємо по днях
            for date, row in data.iterrows():
                if date not in predictions.index:
                    continue
                
                # Поточна ціна та сигнал
                price = row['close'] if 'close' in row else row.iloc[0]
                signal = predictions.loc[date]
                volume = row.get('volume', 1000000)  # Дефолтний обсяг
                volatility = row.get('volatility', 0.02)  # Дефолтна волатильність
                
                # Поточна позиція
                current_position = positions.get('ticker', 0)
                
                # Симуляція торгівлі
                trade_result = self.simulate_trade_execution(
                    signal, price, volume, volatility, current_position
                )
                
                if trade_result['executed']:
                    # Виконуємо торгівлю
                    trade_size = trade_result['trade_size']
                    execution_price = trade_result['execution_price']
                    costs = trade_result['transaction_costs']
                    
                    # Оновлюємо позицію
                    new_position = trade_result['new_position']
                    positions['ticker'] = new_position
                    
                    # Розраховуємо PnL
                    if current_position != 0:
                        pnl = (execution_price - price) * current_position - costs['total']
                        portfolio_value += pnl
                    
                    # Зберігаємо торгівлю
                    trades.append({
                        'date': date,
                        'signal': signal,
                        'trade_size': trade_size,
                        'execution_price': execution_price,
                        'costs': costs['total'],
                        'position': new_position,
                        'portfolio_value': portfolio_value
                    })
                
                # Розраховуємо денну доходність
                if len(trades) > 0:
                    prev_value = trades[-2]['portfolio_value'] if len(trades) > 1 else 1000000
                    daily_return = (portfolio_value - prev_value) / prev_value
                    daily_returns.append(daily_return)
            
            # Розраховуємо метрики продуктивності
            if daily_returns:
                returns_series = pd.Series(daily_returns)
                performance = calculate_performance_metrics(returns_series)
                
                # Загальні витрати
                total_costs = sum(trade['costs'] for trade in trades)
                total_turnover = sum(abs(trade['trade_size']) for trade in trades)
                
                result = {
                    'performance': performance,
                    'trades': trades,
                    'daily_returns': daily_returns,
                    'total_costs': total_costs,
                    'total_turnover': total_turnover,
                    'cost_bps': total_costs / total_turnover * 10000 if total_turnover > 0 else 0,
                    'num_trades': len(trades)
                }
                
                return result
            else:
                return {'performance': {}, 'trades': [], 'total_costs': 0}
                
        except Exception as e:
            self.logger.error(f"Error in period backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_performance(self, periods: List[Dict]) -> Dict[str, any]:
        """
        Розрахувати загальну продуктивність по всіх періодах
        """
        try:
            # Об'єднуємо всі денні доходності
            all_returns = []
            all_costs = 0
            all_turnover = 0
            
            for period in periods:
                if 'daily_returns' in period:
                    all_returns.extend(period['daily_returns'])
                all_costs += period.get('total_costs', 0)
                all_turnover += period.get('total_turnover', 0)
            
            if all_returns:
                returns_series = pd.Series(all_returns)
                overall_performance = calculate_performance_metrics(returns_series)
                
                overall_performance.update({
                    'total_transaction_costs': all_costs,
                    'total_turnover': all_turnover,
                    'average_cost_bps': all_costs / all_turnover * 10000 if all_turnover > 0 else 0,
                    'num_periods': len(periods)
                })
                
                return overall_performance
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calculating overall performance: {e}")
            return {}
    
    def monte_carlo_simulation(self, data: pd.DataFrame, signals: pd.DataFrame,
                             num_simulations: int = 1000) -> Dict[str, any]:
        """
        Монте-Карло симуляція для оцінки надійності стратегії
        
        Args:
            data: Дані
            signals: Сигнали
            num_simulations: Кількість симуляцій
            
        Returns:
            Dict: Результати симуляції
        """
        try:
            simulation_results = []
            
            for sim in range(num_simulations):
                # Випадково перемішуємо сигнали
                shuffled_signals = signals.sample(frac=1, random_state=sim)
                shuffled_signals.index = signals.index
                
                # Бектест з перемішаними сигналами
                sim_result = self._backtest_period(data, shuffled_signals, signals)
                
                if sim_result.get('performance'):
                    sim_result['simulation_id'] = sim
                    simulation_results.append(sim_result)
            
            # Аналізуємо результати
            if simulation_results:
                # Збираємо метрики
                sharpe_ratios = [r['performance'].get('sharpe_ratio', 0) for r in simulation_results]
                total_returns = [r['performance'].get('total_return', 0) for r in simulation_results]
                max_drawdowns = [r['performance'].get('max_drawdown', 0) for r in simulation_results]
                
                # Розраховуємо статистику
                results = {
                    'num_simulations': len(simulation_results),
                    'sharpe_ratio_stats': {
                        'mean': np.mean(sharpe_ratios),
                        'std': np.std(sharpe_ratios),
                        'min': np.min(sharpe_ratios),
                        'max': np.max(sharpe_ratios),
                        'percentile_5': np.percentile(sharpe_ratios, 5),
                        'percentile_95': np.percentile(sharpe_ratios, 95)
                    },
                    'total_return_stats': {
                        'mean': np.mean(total_returns),
                        'std': np.std(total_returns),
                        'min': np.min(total_returns),
                        'max': np.max(total_returns),
                        'percentile_5': np.percentile(total_returns, 5),
                        'percentile_95': np.percentile(total_returns, 95)
                    },
                    'max_drawdown_stats': {
                        'mean': np.mean(max_drawdowns),
                        'std': np.std(max_drawdowns),
                        'min': np.min(max_drawdowns),
                        'max': np.max(max_drawdowns),
                        'percentile_5': np.percentile(max_drawdowns, 5),
                        'percentile_95': np.percentile(max_drawdowns, 95)
                    },
                    'probability_positive_sharpe': np.mean(np.array(sharpe_ratios) > 0),
                    'probability_positive_return': np.mean(np.array(total_returns) > 0)
                }
                
                self.logger.info(f"Monte Carlo simulation completed: {num_simulations} simulations")
                
                return results
            else:
                return {'error': 'No successful simulations'}
                
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}
    
    def stress_test_scenarios(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, any]:
        """
        Стрес-тестинг для різних сценаріїв ринку
        
        Args:
            data: Дані
            signals: Сигнали
            
        Returns:
            Dict: Результати стрес-тестів
        """
        try:
            scenarios = {
                'market_crash': {'price_shock': -0.3, 'volatility_multiplier': 2.0},
                'volatility_spike': {'price_shock': 0, 'volatility_multiplier': 3.0},
                'liquidity_crisis': {'price_shock': -0.1, 'volume_multiplier': 0.3},
                'stagnation': {'price_shock': 0, 'volatility_multiplier': 0.5, 'volume_multiplier': 0.7}
            }
            
            stress_results = {}
            
            for scenario_name, scenario_params in scenarios.items():
                # Модифікуємо дані відповідно до сценарію
                stressed_data = data.copy()
                
                # Ціновий шок
                if 'price_shock' in scenario_params:
                    price_shock = scenario_params['price_shock']
                    stressed_data['close'] = stressed_data['close'] * (1 + price_shock)
                
                # Волатильність
                if 'volatility_multiplier' in scenario_params:
                    vol_multiplier = scenario_params['volatility_multiplier']
                    stressed_data['volatility'] = stressed_data.get('volatility', 0.02) * vol_multiplier
                
                # Обсяг
                if 'volume_multiplier' in scenario_params:
                    vol_multiplier = scenario_params['volume_multiplier']
                    stressed_data['volume'] = stressed_data.get('volume', 1000000) * vol_multiplier
                
                # Бектест на стресованих data
                scenario_result = self._backtest_period(stressed_data, signals, signals)
                
                stress_results[scenario_name] = {
                    'scenario_params': scenario_params,
                    'performance': scenario_result.get('performance', {}),
                    'total_costs': scenario_result.get('total_costs', 0),
                    'num_trades': scenario_result.get('num_trades', 0)
                }
            
            # Порівняння з базовим сценарієм
            baseline_result = self._backtest_period(data, signals, signals)
            
            stress_results['baseline'] = {
                'performance': baseline_result.get('performance', {}),
                'total_costs': baseline_result.get('total_costs', 0),
                'num_trades': baseline_result.get('num_trades', 0)
            }
            
            # Розраховуємо деградацію
            for scenario in scenarios:
                if scenario in stress_results and 'baseline' in stress_results:
                    baseline_sharpe = stress_results['baseline']['performance'].get('sharpe_ratio', 0)
                    scenario_sharpe = stress_results[scenario]['performance'].get('sharpe_ratio', 0)
                    
                    if baseline_sharpe != 0:
                        degradation = (scenario_sharpe - baseline_sharpe) / abs(baseline_sharpe)
                        stress_results[scenario]['sharpe_degradation'] = degradation
            
            self.logger.info(f"Stress testing completed: {len(scenarios)} scenarios")
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Error in stress testing: {e}")
            return {'error': str(e)}
    
    def generate_backtest_report(self, results: Dict[str, any]) -> str:
        """
        Згенерувати звіт бектесту
        
        Args:
            results: Результати бектесту
            
        Returns:
            str: Звіт
        """
        try:
            report = []
            report.append("=" * 60)
            report.append("ADVANCED BACKTESTING REPORT")
            report.append("=" * 60)
            
            # Загальна продуктивність
            if 'overall_performance' in results:
                perf = results['overall_performance']
                report.append("\n[DATA] OVERALL PERFORMANCE:")
                report.append(f"   Total Return: {perf.get('total_return', 0):.2%}")
                report.append(f"   Annual Return: {perf.get('annual_return', 0):.2%}")
                report.append(f"   Annual Volatility: {perf.get('annual_volatility', 0):.2%}")
                report.append(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
                report.append(f"   Sortino Ratio: {perf.get('sortino_ratio', 0):.3f}")
                report.append(f"   Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                report.append(f"   Calmar Ratio: {perf.get('calmar_ratio', 0):.3f}")
            
            # Транзакційні витрати
            if 'average_transaction_costs' in results:
                report.append(f"\n[MONEY] TRANSACTION COSTS:")
                report.append(f"   Average Costs (bps): {results['average_transaction_costs']:.2f}")
                report.append(f"   Total Costs: ${results['overall_performance'].get('total_transaction_costs', 0):,.0f}")
            
            # Walk-forward результати
            if 'periods' in results:
                report.append(f"\n[RESTART] WALK-FORWARD ANALYSIS:")
                report.append(f"   Number of Periods: {len(results['periods'])}")
                report.append(f"   Success Rate: {results.get('success_rate', 0):.1%}")
                
                # Найкращий/найгірший період
                period_sharpes = [p.get('performance', {}).get('sharpe_ratio', 0) for p in results['periods']]
                if period_sharpes:
                    best_period_idx = np.argmax(period_sharpes)
                    worst_period_idx = np.argmin(period_sharpes)
                    
                    report.append(f"   Best Period Sharpe: {period_sharpes[best_period_idx]:.3f}")
                    report.append(f"   Worst Period Sharpe: {period_sharpes[worst_period_idx]:.3f}")
            
            # Монте-Карло результати
            if 'sharpe_ratio_stats' in results:
                mc = results
                report.append(f"\n🎰 MONTE CARLO SIMULATION:")
                report.append(f"   Simulations: {mc.get('num_simulations', 0)}")
                report.append(f"   Sharpe Ratio - Mean: {mc['sharpe_ratio_stats']['mean']:.3f}")
                report.append(f"   Sharpe Ratio - Std: {mc['sharpe_ratio_stats']['std']:.3f}")
                report.append(f"   Prob. Positive Sharpe: {mc.get('probability_positive_sharpe', 0):.1%}")
            
            # Стрес-тест результати
            if 'baseline' in results:
                report.append(f"\n🚨 STRESS TESTING:")
                baseline_sharpe = results['baseline']['performance'].get('sharpe_ratio', 0)
                report.append(f"   Baseline Sharpe: {baseline_sharpe:.3f}")
                
                for scenario in ['market_crash', 'volatility_spike', 'liquidity_crisis']:
                    if scenario in results:
                        scenario_sharpe = results[scenario]['performance'].get('sharpe_ratio', 0)
                        degradation = results[scenario].get('sharpe_degradation', 0)
                        report.append(f"   {scenario.title()}: {scenario_sharpe:.3f} ({degradation:+.1%})")
            
            report.append("\n" + "=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return "Error generating report"


# Глобальний екземпляр
advanced_backtester = AdvancedBacktester()


def run_walk_forward_analysis(data: pd.DataFrame, signals: pd.DataFrame,
                           train_model_func, predict_func) -> Dict[str, any]:
    """Запустити walk-forward аналіз"""
    return advanced_backtester.walk_forward_analysis(data, signals, train_model_func, predict_func)


def run_monte_carlo_simulation(data: pd.DataFrame, signals: pd.DataFrame,
                              num_simulations: int = 1000) -> Dict[str, any]:
    """Запустити Монте-Карло симуляцію"""
    return advanced_backtester.monte_carlo_simulation(data, signals, num_simulations)


def run_stress_test_scenarios(data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, any]:
    """Запустити стрес-тестинг"""
    return advanced_backtester.stress_test_scenarios(data, signals)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Advanced Backtester Test")
    print("="*50)
    
    # Симуляція data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Ціни та обсяги
    data = pd.DataFrame(index=dates)
    data['close'] = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
    data['volume'] = np.random.lognormal(15, 0.5, len(dates))
    data['volatility'] = np.random.uniform(0.01, 0.04, len(dates))
    
    # Сигнали
    signals = pd.DataFrame(index=dates)
    signals['signal'] = np.random.uniform(-1, 1, len(dates))
    
    # Проста модель для тестування
    def simple_train_model(train_data, train_signals, **params):
        return {'mean_signal': train_signals['signal'].mean()}
    
    def simple_predict(model, test_data):
        mean_signal = model['mean_signal']
        return pd.Series([mean_signal] * len(test_data), index=test_data.index)
    
    # Walk-forward аналіз
    wf_results = run_walk_forward_analysis(data, signals, simple_train_model, simple_predict)
    
    if 'overall_performance' in wf_results:
        print(f"[DATA] Walk-Forward Results:")
        perf = wf_results['overall_performance']
        print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        print(f"   Total Return: {perf.get('total_return', 0):.2%}")
        print(f"   Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        print(f"   Success Rate: {wf_results.get('success_rate', 0):.1%}")
        
        # Монте-Карло
        mc_results = run_monte_carlo_simulation(data, signals, num_simulations=100)
        if 'sharpe_ratio_stats' in mc_results:
            print(f"\n🎰 Monte Carlo Results:")
            mc = mc_results['sharpe_ratio_stats']
            print(f"   Mean Sharpe: {mc['mean']:.3f}")
            print(f"   Std Sharpe: {mc['std']:.3f}")
            print(f"   Prob. Positive: {mc_results.get('probability_positive_sharpe', 0):.1%}")
        
        # Стрес-тест
        stress_results = run_stress_test_scenarios(data, signals)
        if 'baseline' in stress_results:
            print(f"\n🚨 Stress Test Results:")
            baseline = stress_results['baseline']['performance'].get('sharpe_ratio', 0)
            print(f"   Baseline Sharpe: {baseline:.3f}")
            
            for scenario in ['market_crash', 'volatility_spike']:
                if scenario in stress_results:
                    scenario_sharpe = stress_results[scenario]['performance'].get('sharpe_ratio', 0)
                    print(f"   {scenario.title()}: {scenario_sharpe:.3f}")
        
        # Звіт
        report = advanced_backtester.generate_backtest_report(wf_results)
        print(f"\n{report}")
        
        print(f"\n[OK] Advanced Backtesting working correctly!")
    else:
        print(f"[ERROR] Walk-forward analysis failed")
