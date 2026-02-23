#!/usr/bin/env python3
"""
Backtest mode - повноцінний бектестинг стратегій
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json

from .base import BaseMode
from config.trading_config import TradingConfig


class BacktestMode(BaseMode):
    """Режим повноцінного бектестингу торгових стратегій"""
    
    def run(self) -> Dict[str, Any]:
        """Запуск бектестингу"""
        self.logger.info("Starting comprehensive backtesting mode...")
        
        try:
            # Перевірка передумов
            if not self.validate_prerequisites():
                return {
                    'status': 'failed',
                    'error': 'Prerequisites validation failed',
                    'message': 'Cannot start backtesting - missing requirements'
                }
            
            # Ініціалізація компонентів
            self._initialize_backtest_components()
            
            # Завантаження історичних data
            self.logger.info("Loading historical data...")
            historical_data = self._load_historical_data()
            
            # Ініціалізація стратегій
            self.logger.info("Initializing trading strategies...")
            strategies = self._initialize_strategies()
            
            # Запуск бектестингу
            self.logger.info("Running backtest simulations...")
            backtest_results = self._run_backtest_simulations(historical_data, strategies)
            
            # Аналіз результатів
            self.logger.info("Analyzing backtest results...")
            analysis_results = self._analyze_backtest_results(backtest_results)
            
            # Генерація звітів
            self.logger.info("Generating comprehensive reports...")
            reports = self._generate_reports(backtest_results, analysis_results)
            
            # Збереження результатів
            results = self._save_backtest_results(backtest_results, analysis_results, reports)
            
            self.logger.info("Backtesting completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Backtesting process failed'
            }
        finally:
            self.cleanup()
    
    def validate_prerequisites(self) -> bool:
        """Перевірка передумов для бектестингу"""
        if not self.config.data.tickers:
            self.logger.error("No tickers configured for backtesting")
            return False
        
        if not self.config.data.timeframes:
            self.logger.error("No timeframes configured for backtesting")
            return False
        
        if self.config.risk.max_positions <= 0:
            self.logger.error("Invalid max positions configuration")
            return False
        
        return True
    
    def _initialize_backtest_components(self) -> None:
        """Ініціалізація компонентів для бектестингу"""
        self.logger.info("Initializing backtesting components...")
        
        # Ініціалізація портфеля
        self.portfolio = {
            'cash': 100000.0,  # Початковий капітал
            'positions': {},
            'transactions': [],
            'portfolio_value': []
        }
        
        # Налаштування бектестингу
        self.backtest_config = {
            'start_date': datetime.now() - timedelta(days=365),
            'end_date': datetime.now(),
            'commission': 0.001,  # 0.1% комісія
            'slippage': 0.0001,   # 0.01% slippage
            'initial_capital': 100000.0
        }
        
        self.logger.info(f"Backtest period: {self.backtest_config['start_date']} to {self.backtest_config['end_date']}")
    
    def _load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Завантаження історичних data для всіх тікерів"""
        historical_data = {}
        
        for ticker in self.config.data.tickers:
            try:
                # Симуляція завантаження data (в реальності - з API або файлів)
                dates = pd.date_range(
                    start=self.backtest_config['start_date'],
                    end=self.backtest_config['end_date'],
                    freq='D'
                )
                
                # Генерація симуляційних OHLCV data
                np.random.seed(hash(ticker) % 1000)
                price_base = np.random.uniform(50, 500)
                
                data = pd.DataFrame({
                    'date': dates,
                    'open': price_base * (1 + np.random.normal(0, 0.02, len(dates)).cumsum() * 0.001),
                    'high': price_base * (1 + np.random.normal(0, 0.03, len(dates)).cumsum() * 0.001),
                    'low': price_base * (1 + np.random.normal(0, 0.03, len(dates)).cumsum() * 0.001),
                    'close': price_base * (1 + np.random.normal(0, 0.02, len(dates)).cumsum() * 0.001),
                    'volume': np.random.randint(100000, 10000000, len(dates))
                })
                
                # Виправлення OHLC логіки
                data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
                data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
                
                historical_data[ticker] = data
                self.logger.info(f"Loaded {len(data)} data points for {ticker}")
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {ticker}: {e}")
                continue
        
        return historical_data
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Ініціалізація торгових стратегій"""
        strategies = {
            'momentum': {
                'name': 'Momentum Strategy',
                'description': 'Buy on positive momentum, sell on negative',
                'parameters': {
                    'lookback_period': 20,
                    'momentum_threshold': 0.02,
                    'position_size': 0.1
                }
            },
            'mean_reversion': {
                'name': 'Mean Reversion Strategy',
                'description': 'Buy when price is below mean, sell when above',
                'parameters': {
                    'lookback_period': 50,
                    'deviation_threshold': 2.0,
                    'position_size': 0.1
                }
            },
            'sentiment': {
                'name': 'Sentiment Strategy',
                'description': 'Trade based on sentiment analysis',
                'parameters': {
                    'sentiment_threshold': 0.1,
                    'position_size': 0.15
                }
            },
            'volatility': {
                'name': 'Volatility Strategy',
                'description': 'Trade based on volatility patterns',
                'parameters': {
                    'volatility_window': 20,
                    'volatility_threshold': 0.03,
                    'position_size': 0.08
                }
            }
        }
        
        self.logger.info(f"Initialized {len(strategies)} trading strategies")
        return strategies
    
    def _run_backtest_simulations(self, historical_data: Dict[str, pd.DataFrame], 
                                 strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Запуск симуляцій бектестингу для всіх стратегій"""
        results = {}
        
        for strategy_name, strategy_config in strategies.items():
            self.logger.info(f"Running backtest for {strategy_name} strategy...")
            
            try:
                # Скидання портфеля для кожної стратегії
                self._reset_portfolio()
                
                # Запуск стратегії
                strategy_results = self._run_single_strategy(
                    historical_data, 
                    strategy_name, 
                    strategy_config
                )
                
                results[strategy_name] = strategy_results
                self.logger.info(f"Completed {strategy_name}: {strategy_results.get('metrics', {}).get('total_return', 0):.2%} return")
                
            except Exception as e:
                self.logger.error(f"Failed to run {strategy_name}: {e}")
                results[strategy_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _run_single_strategy(self, historical_data: Dict[str, pd.DataFrame], 
                            strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск однієї стратегії"""
        portfolio_values = [self.backtest_config['initial_capital']]
        trades = []
        daily_returns = []
        
        # Отримання всіх дат з data
        all_dates = set()
        for ticker_data in historical_data.values():
            all_dates.update(ticker_data['date'])
        
        sorted_dates = sorted(all_dates)
        
        for current_date in sorted_dates:
            # Оновлення портфеля
            self._update_portfolio_value(current_date, historical_data)
            portfolio_values.append(self.portfolio['total_value'])
            
            # Генерація сигналів
            signals = self._generate_signals(current_date, historical_data, strategy_name, strategy_config)
            
            # Виконання торгів
            new_trades = self._execute_trades(current_date, signals, historical_data, strategy_config)
            trades.extend(new_trades)
            
            # Розрахунок денного доходу
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)
        
        # Розрахунок метрик
        metrics = self._calculate_performance_metrics(portfolio_values, daily_returns, trades)
        
        return {
            'status': 'success',
            'strategy_name': strategy_name,
            'portfolio_values': portfolio_values,
            'trades': trades,
            'daily_returns': daily_returns,
            'metrics': metrics
        }
    
    def _reset_portfolio(self) -> None:
        """Скидання портфеля до початкового стану"""
        self.portfolio = {
            'cash': self.backtest_config['initial_capital'],
            'positions': {},
            'transactions': [],
            'total_value': self.backtest_config['initial_capital']
        }
    
    def _update_portfolio_value(self, date: datetime, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Оновлення вартості портфеля"""
        total_value = self.portfolio['cash']
        
        for ticker, position in self.portfolio['positions'].items():
            if ticker in historical_data:
                ticker_data = historical_data[ticker]
                current_price_data = ticker_data[ticker_data['date'] == date]
                
                if not current_price_data.empty:
                    current_price = current_price_data['close'].iloc[0]
                    position_value = position['shares'] * current_price
                    total_value += position_value
        
        self.portfolio['total_value'] = total_value
    
    def _generate_signals(self, date: datetime, historical_data: Dict[str, pd.DataFrame], 
                         strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, str]:
        """Генерація торгових сигналів"""
        signals = {}
        
        for ticker in self.config.data.tickers:
            if ticker not in historical_data:
                continue
            
            ticker_data = historical_data[ticker]
            current_data = ticker_data[ticker_data['date'] <= date]
            
            if len(current_data) < 50:  # Мінімальна кількість data
                continue
            
            # Генерація сигналів залежно від стратегії
            if strategy_name == 'momentum':
                signal = self._generate_momentum_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'mean_reversion':
                signal = self._generate_mean_reversion_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'sentiment':
                signal = self._generate_sentiment_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'volatility':
                signal = self._generate_volatility_signal(current_data, strategy_config['parameters'])
            else:
                signal = 'HOLD'
            
            signals[ticker] = signal
        
        return signals
    
    def _generate_momentum_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Генерація сигналу для momentum стратегії"""
        lookback = params['lookback_period']
        threshold = params['momentum_threshold']
        
        if len(data) < lookback:
            return 'HOLD'
        
        recent_prices = data['close'].tail(lookback)
        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if momentum > threshold:
            return 'BUY'
        elif momentum < -threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_mean_reversion_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Генерація сигналу для mean reversion стратегії"""
        lookback = params['lookback_period']
        threshold = params['deviation_threshold']
        
        if len(data) < lookback:
            return 'HOLD'
        
        recent_prices = data['close'].tail(lookback)
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        current_price = recent_prices.iloc[-1]
        
        z_score = (current_price - mean_price) / std_price
        
        if z_score < -threshold:
            return 'BUY'
        elif z_score > threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_sentiment_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Генерація сигналу для sentiment стратегії"""
        # Симуляція sentiment аналізу
        threshold = params['sentiment_threshold']
        
        # Симуляція sentiment score (-1 до 1)
        np.random.seed(len(data))
        sentiment_score = np.random.normal(0, 0.3)
        
        if sentiment_score > threshold:
            return 'BUY'
        elif sentiment_score < -threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_volatility_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Генерація сигналу для volatility стратегії"""
        window = params['volatility_window']
        threshold = params['volatility_threshold']
        
        if len(data) < window:
            return 'HOLD'
        
        returns = data['close'].pct_change().tail(window)
        volatility = returns.std()
        
        # Симуляція volatility trading
        if volatility > threshold:
            return 'BUY'  # Купувати при високій волатильності
        else:
            return 'HOLD'
    
    def _execute_trades(self, date: datetime, signals: Dict[str, str], 
                       historical_data: Dict[str, pd.DataFrame], 
                       strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Виконання торгів на основі сигналів"""
        trades = []
        position_size = strategy_config['parameters']['position_size']
        
        for ticker, signal in signals.items():
            if ticker not in historical_data:
                continue
            
            ticker_data = historical_data[ticker]
            current_price_data = ticker_data[ticker_data['date'] == date]
            
            if current_price_data.empty:
                continue
            
            current_price = current_price_data['close'].iloc[0]
            
            if signal == 'BUY' and ticker not in self.portfolio['positions']:
                # Купівля
                available_cash = self.portfolio['cash'] * position_size
                shares_to_buy = int(available_cash / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.backtest_config['commission'])
                    
                    if cost <= self.portfolio['cash']:
                        self.portfolio['cash'] -= cost
                        self.portfolio['positions'][ticker] = {
                            'shares': shares_to_buy,
                            'entry_price': current_price,
                            'entry_date': date
                        }
                        
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost
                        })
            
            elif signal == 'SELL' and ticker in self.portfolio['positions']:
                # Продаж
                position = self.portfolio['positions'][ticker]
                shares_to_sell = position['shares']
                
                if shares_to_sell > 0:
                    revenue = shares_to_sell * current_price * (1 - self.backtest_config['commission'])
                    
                    self.portfolio['cash'] += revenue
                    del self.portfolio['positions'][ticker]
                    
                    trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'revenue': revenue,
                        'profit': revenue - (shares_to_sell * position['entry_price'])
                    })
        
        return trades
    
    def _calculate_performance_metrics(self, portfolio_values: List[float], 
                                    daily_returns: List[float], 
                                    trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Розрахунок метрик продуктивності"""
        if not portfolio_values:
            return {}
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Розрахунок додаткових метрик
        metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t.get('profit', 0) > 0]),
            'losing_trades': len([t for t in trades if t.get('profit', 0) < 0])
        }
        
        if daily_returns:
            metrics.update({
                'daily_return_mean': np.mean(daily_returns),
                'daily_return_std': np.std(daily_returns),
                'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'volatility': np.std(daily_returns) * np.sqrt(252)
            })
        
        if trades:
            profits = [t.get('profit', 0) for t in trades if 'profit' in t]
            metrics.update({
                'average_profit_per_trade': np.mean(profits) if profits else 0,
                'profit_factor': sum(p for p in profits if p > 0) / abs(sum(p for p in profits if p < 0)) if any(p < 0 for p in profits) else float('inf')
            })
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Розрахунок максимального drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _analyze_backtest_results(self, backtest_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Аналіз результатів бектестингу"""
        analysis = {
            'summary': {},
            'best_strategy': None,
            'worst_strategy': None,
            'strategy_comparison': {},
            'risk_analysis': {}
        }
        
        successful_results = {k: v for k, v in backtest_results.items() if v.get('status') == 'success'}
        
        if not successful_results:
            return analysis
        
        # Порівняння стратегій
        returns = {}
        sharpe_ratios = {}
        max_drawdowns = {}
        
        for strategy_name, result in successful_results.items():
            metrics = result.get('metrics', {})
            returns[strategy_name] = metrics.get('total_return', 0)
            sharpe_ratios[strategy_name] = metrics.get('sharpe_ratio', 0)
            max_drawdowns[strategy_name] = metrics.get('max_drawdown', 0)
        
        # Найкраща стратегія за return
        if returns:
            best_strategy = max(returns, key=returns.get)
            worst_strategy = min(returns, key=returns.get)
            
            analysis['best_strategy'] = {
                'name': best_strategy,
                'return': returns[best_strategy],
                'sharpe_ratio': sharpe_ratios.get(best_strategy, 0),
                'max_drawdown': max_drawdowns.get(best_strategy, 0)
            }
            
            analysis['worst_strategy'] = {
                'name': worst_strategy,
                'return': returns[worst_strategy],
                'sharpe_ratio': sharpe_ratios.get(worst_strategy, 0),
                'max_drawdown': max_drawdowns.get(worst_strategy, 0)
            }
        
        # Статистичний аналіз
        if returns:
            analysis['summary'] = {
                'total_strategies': len(successful_results),
                'average_return': np.mean(list(returns.values())),
                'return_std': np.std(list(returns.values())),
                'positive_return_strategies': len([r for r in returns.values() if r > 0]),
                'negative_return_strategies': len([r for r in returns.values() if r < 0])
            }
        
        return analysis
    
    def _generate_reports(self, backtest_results: Dict[str, Dict[str, Any]], 
                         analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерація звітів"""
        reports = {
            'executive_summary': self._generate_executive_summary(analysis_results),
            'detailed_results': backtest_results,
            'risk_analysis': self._generate_risk_analysis(backtest_results),
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        return reports
    
    def _generate_executive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерація виконавчого звіту"""
        summary = analysis_results.get('summary', {})
        best_strategy = analysis_results.get('best_strategy', {})
        
        return {
            'total_strategies_tested': summary.get('total_strategies', 0),
            'best_performing_strategy': best_strategy.get('name', 'N/A'),
            'best_strategy_return': f"{best_strategy.get('return', 0):.2%}",
            'average_return_all_strategies': f"{summary.get('average_return', 0):.2%}",
            'successful_strategies': summary.get('positive_return_strategies', 0),
            'unsuccessful_strategies': summary.get('negative_return_strategies', 0)
        }
    
    def _generate_risk_analysis(self, backtest_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Генерація аналізу ризиків"""
        risk_metrics = {}
        
        for strategy_name, result in backtest_results.items():
            if result.get('status') == 'success':
                metrics = result.get('metrics', {})
                risk_metrics[strategy_name] = {
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'volatility': metrics.get('volatility', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_trades': metrics.get('total_trades', 0)
                }
        
        return risk_metrics
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Генерація рекомендацій"""
        recommendations = []
        
        best_strategy = analysis_results.get('best_strategy')
        summary = analysis_results.get('summary', {})
        
        if best_strategy:
            recommendations.append(f"Consider allocating more capital to {best_strategy['name']} strategy")
        
        if summary.get('positive_return_strategies', 0) > summary.get('negative_return_strategies', 0):
            recommendations.append("Majority of strategies are profitable - consider portfolio diversification")
        else:
            recommendations.append("Most strategies are unprofitable - review strategy parameters")
        
        if best_strategy and best_strategy.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("Low Sharpe ratios detected - consider risk-adjusted improvements")
        
        return recommendations
    
    def _save_backtest_results(self, backtest_results: Dict[str, Dict[str, Any]], 
                             analysis_results: Dict[str, Any], 
                             reports: Dict[str, Any]) -> Dict[str, Any]:
        """Збереження результатів бектестингу"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.config.data.data_dir / 'results' / 'backtest'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Основний файл результатів
        results_file = results_dir / f"backtest_{timestamp}.json"
        
        final_results = {
            'status': 'success',
            'mode': 'backtest',
            'timestamp': timestamp,
            'config': {
                'tickers': self.config.data.tickers,
                'timeframes': self.config.data.timeframes,
                'backtest_period': {
                    'start': self.backtest_config['start_date'].isoformat(),
                    'end': self.backtest_config['end_date'].isoformat()
                },
                'initial_capital': self.backtest_config['initial_capital']
            },
            'backtest_results': backtest_results,
            'analysis_results': analysis_results,
            'reports': reports
        }
        
        # Збереження в файл
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"Backtest results saved to {results_file}")
        
        return final_results
    
    def cleanup(self) -> None:
        """Очищення ресурсів"""
        self.logger.info("Cleaning up backtesting resources...")
        # Очищення великих об'єктів data
        if hasattr(self, 'historical_data'):
            del self.historical_data
        
        if hasattr(self, 'portfolio'):
            del self.portfolio
