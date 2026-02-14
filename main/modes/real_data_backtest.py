#!/usr/bin/env python3
"""
Інтегрований режим бектестингу з реальними даними
Поєднує всі покращення та використовує реальні ринкові дані
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from .base import BaseMode
from config.trading_config import TradingConfig
from utils.real_data_collector import RealDataCollector
from utils.enhanced_data_validator import DataValidator
from utils.common_utils import CacheManager, PerformanceMonitor, TechnicalIndicators, ErrorHandler


class RealDataBacktestMode(BaseMode):
    """Режим бектестингу з реальними даними"""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.data_collector = RealDataCollector(config)
        self.validator = DataValidator()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        self.technical_indicators = TechnicalIndicators()
        
        # Конфігурація бектестингу
        self.backtest_config = {
            'start_date': datetime.now() - timedelta(days=365),
            'end_date': datetime.now(),
            'commission': 0.001,
            'slippage': 0.0001,
            'initial_capital': self.config.risk.initial_capital,
            'use_real_data': True,
            'data_source_priority': ['yahoo_finance', 'alpha_vantage', 'finnhub']
        }
        
        # Ініціалізація портфоліо
        self.portfolio = {
            'cash': self.backtest_config['initial_capital'],
            'positions': {},
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {}
        }
    
    def validate_prerequisites(self) -> bool:
        """Перевірка передумов"""
        if not self.config.data.tickers:
            self.logger.error("No tickers configured for backtesting")
            return False
        
        if not self.config.data.timeframes:
            self.logger.error("No timeframes configured for backtesting")
            return False
        
        if self.config.risk.max_positions <= 0:
            self.logger.error("Invalid max positions configuration")
            return False
        
        # Перевірка API keysв
        has_api_keys = (
            hasattr(self.config, 'alpha_vantage_key') and self.config.alpha_vantage_key or
            hasattr(self.config, 'finnhub_key') and self.config.finnhub_key
        )
        
        if not has_api_keys:
            self.logger.warning("No API keys configured. Will use Yahoo Finance only.")
        
        return True
    
    def _load_real_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Завантаження реальних історичних data"""
        return self.performance_monitor.time_execution("load_real_historical_data")(self._load_real_historical_data_impl)()
    
    def _load_real_historical_data_impl(self) -> Dict[str, pd.DataFrame]:
        """Завантаження реальних історичних data"""
        self.logger.info("Loading real historical data...")
        
        historical_data = {}
        
        # Визначення періоду та інтервалу
        days_back = (self.backtest_config['end_date'] - self.backtest_config['start_date']).days
        period = f"{days_back}d" if days_back <= 365 else "1y"
        interval = "1d"  # Щодня для історичних data
        
        # Масове завантаження data
        data = self.data_collector.collect_batch_data(
            self.config.data.tickers, 
            period=period, 
            interval=interval
        )
        
        # Фільтрація data по даті
        for ticker, df in data.items():
            if df is not None and not df.empty:
                # Конвертація дати та фільтрація
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[
                        (df['date'] >= self.backtest_config['start_date']) &
                        (df['date'] <= self.backtest_config['end_date'])
                    ]
                    df = df.sort_values('date').reset_index(drop=True)
                
                # Валідація та очищення data
                try:
                    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    df = self.validator.sanitize_dataframe(df, required_columns)
                    historical_data[ticker] = df
                    self.logger.info(f"Loaded {len(df)} records for {ticker}")
                except Exception as e:
                    self.logger.warning(f"Data validation failed for {ticker}: {e}")
                    # Продовжуємо з невалідованими даними
                    historical_data[ticker] = df
        
        if not historical_data:
            raise ValueError("No valid historical data loaded")
        
        self.logger.info(f"Successfully loaded data for {len(historical_data)} tickers")
        return historical_data
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Ініціалізація торгових стратегій"""
        strategies = {
            'momentum': {
                'name': 'Momentum Strategy',
                'parameters': {
                    'lookback_period': 20,
                    'momentum_threshold': 0.02
                }
            },
            'mean_reversion': {
                'name': 'Mean Reversion Strategy',
                'parameters': {
                    'lookback_period': 20,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.5
                }
            },
            'rsi': {
                'name': 'RSI Strategy',
                'parameters': {
                    'period': 14,
                    'oversold': 30,
                    'overbought': 70
                }
            },
            'macd': {
                'name': 'MACD Strategy',
                'parameters': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9
                }
            },
            'bollinger_bands': {
                'name': 'Bollinger Bands Strategy',
                'parameters': {
                    'period': 20,
                    'std': 2.0
                }
            },
            'volume_price': {
                'name': 'Volume Price Strategy',
                'parameters': {
                    'volume_ma_period': 20,
                    'price_change_threshold': 0.02
                }
            }
        }
        
        self.logger.info(f"Initialized {len(strategies)} strategies")
        return strategies
    
    def _generate_signals(self, date: datetime, historical_data: Dict[str, pd.DataFrame], 
                         strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, str]:
        """Генерація торгових сигналів на основі реальних data"""
        return self.performance_monitor.time_execution("generate_signals")(self._generate_signals_impl)(date, historical_data, strategy_name, strategy_config)
    
    def _generate_signals_impl(self, date: datetime, historical_data: Dict[str, pd.DataFrame], 
                              strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, str]:
        """Генерація торгових сигналів на основі реальних data"""
        signals = {}
        
        for ticker, data in historical_data.items():
            try:
                # Отримання data до поточної дати
                current_data = data[data['date'] <= date]
                if len(current_data) < 50:  # Мінімальна кількість data
                    continue
                
                latest_data = current_data.iloc[-1]
                prices = current_data['close']
                
                # Генерація сигналів залежно від стратегії
                if strategy_name == 'momentum':
                    momentum_period = strategy_config['parameters']['lookback_period']
                    if len(prices) >= momentum_period:
                        momentum = (prices.iloc[-1] / prices.iloc[-momentum_period]) - 1
                        threshold = strategy_config['parameters']['momentum_threshold']
                        
                        if momentum > threshold:
                            signals[ticker] = 'BUY'
                        elif momentum < -threshold:
                            signals[ticker] = 'SELL'
                        else:
                            signals[ticker] = 'HOLD'
                
                elif strategy_name == 'mean_reversion':
                    lookback = strategy_config['parameters']['lookback_period']
                    if len(prices) >= lookback:
                        mean_price = prices.iloc[-lookback:].mean()
                        std_price = prices.iloc[-lookback:].std()
                        z_score = (latest_data['close'] - mean_price) / std_price
                        
                        entry_threshold = strategy_config['parameters']['entry_threshold']
                        exit_threshold = strategy_config['parameters']['exit_threshold']
                        
                        if z_score > entry_threshold:
                            signals[ticker] = 'SELL'
                        elif z_score < -entry_threshold:
                            signals[ticker] = 'BUY'
                        elif abs(z_score) < exit_threshold:
                            signals[ticker] = 'HOLD'
                
                elif strategy_name == 'rsi':
                    period = strategy_config['parameters']['period']
                    if len(prices) >= period:
                        rsi = self.technical_indicators.calculate_rsi(prices, period)
                        current_rsi = rsi.iloc[-1]
                        
                        oversold = strategy_config['parameters']['oversold']
                        overbought = strategy_config['parameters']['overbought']
                        
                        if current_rsi < oversold:
                            signals[ticker] = 'BUY'
                        elif current_rsi > overbought:
                            signals[ticker] = 'SELL'
                        else:
                            signals[ticker] = 'HOLD'
                
                elif strategy_name == 'macd':
                    fast = strategy_config['parameters']['fast']
                    slow = strategy_config['parameters']['slow']
                    signal = strategy_config['parameters']['signal']
                    
                    if len(prices) >= slow:
                        macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(
                            prices, fast, slow, signal
                        )
                        
                        if len(histogram) >= 2:
                            if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
                                signals[ticker] = 'BUY'
                            elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
                                signals[ticker] = 'SELL'
                            else:
                                signals[ticker] = 'HOLD'
                
                elif strategy_name == 'bollinger_bands':
                    period = strategy_config['parameters']['period']
                    std = strategy_config['parameters']['std']
                    
                    if len(prices) >= period:
                        upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(
                            prices, period, std
                        )
                        
                        current_price = latest_data['close']
                        current_upper = upper.iloc[-1]
                        current_lower = lower.iloc[-1]
                        
                        if current_price <= current_lower:
                            signals[ticker] = 'BUY'
                        elif current_price >= current_upper:
                            signals[ticker] = 'SELL'
                        else:
                            signals[ticker] = 'HOLD'
                
                elif strategy_name == 'volume_price':
                    volume_ma_period = strategy_config['parameters']['volume_ma_period']
                    price_change_threshold = strategy_config['parameters']['price_change_threshold']
                    
                    if len(current_data) >= volume_ma_period:
                        volume_ma = current_data['volume'].iloc[-volume_ma_period:].mean()
                        current_volume = latest_data['volume']
                        
                        price_change = (latest_data['close'] / latest_data['open']) - 1
                        
                        if current_volume > volume_ma * 1.5 and abs(price_change) > price_change_threshold:
                            signals[ticker] = 'BUY' if price_change > 0 else 'SELL'
                        else:
                            signals[ticker] = 'HOLD'
                
            except Exception as e:
                self.logger.warning(f"Error generating signal for {ticker}: {e}")
                signals[ticker] = 'HOLD'
        
        return signals
    
    def _execute_trades(self, date: datetime, signals: Dict[str, str], 
                       historical_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Виконання торгів на основі сигналів"""
        trades = []
        
        for ticker, signal in signals.items():
            if ticker not in historical_data:
                continue
            
            ticker_data = historical_data[ticker]
            current_price_data = ticker_data[ticker_data['date'] == date]
            
            if current_price_data.empty:
                continue
            
            current_price = current_price_data['close'].iloc[0]
            
            try:
                position_size = self.config.risk.risk_per_trade
                
                if signal == 'BUY' and ticker not in self.portfolio['positions']:
                    quantity = int(self.portfolio['cash'] * position_size / current_price)
                    if quantity > 0:
                        trade = {
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_price,
                            'value': quantity * current_price,
                            'commission': quantity * current_price * self.backtest_config['commission']
                        }
                        
                        self.portfolio['cash'] -= (trade['value'] + trade['commission'])
                        self.portfolio['positions'][ticker] = {
                            'quantity': quantity,
                            'entry_price': current_price,
                            'entry_date': date
                        }
                        trades.append(trade)
                
                elif signal == 'SELL' and ticker in self.portfolio['positions']:
                    position = self.portfolio['positions'][ticker]
                    quantity = position['quantity']
                    
                    trade = {
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': current_price,
                        'value': quantity * current_price,
                        'commission': quantity * current_price * self.backtest_config['commission']
                    }
                    
                    self.portfolio['cash'] += (trade['value'] - trade['commission'])
                    del self.portfolio['positions'][ticker]
                    trades.append(trade)
                
            except Exception as e:
                self.logger.warning(f"Error executing trade for {ticker}: {e}")
        
        return trades
    
    def _calculate_portfolio_value(self, date: datetime, historical_data: Dict[str, pd.DataFrame]) -> float:
        """Розрахунок вартості портфоліо"""
        total_value = self.portfolio['cash']
        
        for ticker, position in self.portfolio['positions'].items():
            if ticker in historical_data:
                ticker_data = historical_data[ticker]
                current_price_data = ticker_data[ticker_data['date'] == date]
                
                if not current_price_data.empty:
                    current_price = current_price_data['close'].iloc[0]
                    total_value += position['quantity'] * current_price
        
        return total_value
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Розрахунок метрик продуктивності"""
        equity_curve = pd.DataFrame(self.portfolio['equity_curve'])
        
        if equity_curve.empty:
            return {}
        
        # Базові метрики
        initial_value = equity_curve['portfolio_value'].iloc[0]
        final_value = equity_curve['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Щоденна доходність
        equity_curve['daily_return'] = equity_curve['portfolio_value'].pct_change()
        daily_returns = equity_curve['daily_return'].dropna()
        
        # Ризикові метрики
        volatility = daily_returns.std() * np.sqrt(252)  # Річна волатильність
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Максимальна просадка
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Статистика торгів
        trades_df = pd.DataFrame(self.portfolio['trades'])
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['action'] == 'SELL']
            if not winning_trades.empty:
                win_rate = len(winning_trades[winning_trades['value'] > winning_trades['commission']]) / len(winning_trades)
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(equity_curve)),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.portfolio['trades']),
            'final_portfolio_value': final_value,
            'initial_portfolio_value': initial_value
        }
        
        return metrics
    
    def _save_backtest_results(self, results: Dict[str, Any]) -> str:
        """Збереження результатів бектестингу"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_data_backtest_{timestamp}.json"
        
        results_dir = self.config.data.data_dir / 'results' / 'real_data_backtest'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        return str(results_file)
    
    def run(self) -> Dict[str, Any]:
        """Запуск бектестингу з реальними даними"""
        try:
            self.logger.info("Starting real data backtesting...")
            
            # Валідація передумов
            if not self.validate_prerequisites():
                raise ValueError("Prerequisites validation failed")
            
            # Завантаження реальних data
            historical_data = self._load_real_historical_data()
            
            # Ініціалізація стратегій
            strategies = self._initialize_strategies()
            
            # Отримання діапазону дат
            all_dates = set()
            for data in historical_data.values():
                all_dates.update(data['date'].dt.date.tolist())
            
            date_range = sorted(list(all_dates))
            
            if not date_range:
                raise ValueError("No valid dates found in historical data")
            
            # Основний цикл бектестингу
            for date in date_range:
                date_dt = datetime.combine(date, datetime.min.time())
                
                # Генерація сигналів для кожної стратегії
                all_signals = {}
                for strategy_name, strategy_config in strategies.items():
                    signals = self._generate_signals(date_dt, historical_data, strategy_name, strategy_config)
                    all_signals[strategy_name] = signals
                
                # Комбінація сигналів (проста голосування)
                combined_signals = {}
                for ticker in self.config.data.tickers:
                    votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                    for strategy_signals in all_signals.values():
                        if ticker in strategy_signals:
                            votes[strategy_signals[ticker]] += 1
                    
                    # Вибір сигналу з більшістю голосів
                    if votes['BUY'] > votes['SELL'] and votes['BUY'] > votes['HOLD']:
                        combined_signals[ticker] = 'BUY'
                    elif votes['SELL'] > votes['BUY'] and votes['SELL'] > votes['HOLD']:
                        combined_signals[ticker] = 'SELL'
                    else:
                        combined_signals[ticker] = 'HOLD'
                
                # Виконання торгів
                trades = self._execute_trades(date_dt, combined_signals, historical_data)
                self.portfolio['trades'].extend(trades)
                
                # Оновлення вартості портфоліо
                portfolio_value = self._calculate_portfolio_value(date_dt, historical_data)
                self.portfolio['equity_curve'].append({
                    'date': date_dt,
                    'portfolio_value': portfolio_value,
                    'cash': self.portfolio['cash'],
                    'positions_count': len(self.portfolio['positions'])
                })
            
            # Розрахунок метрик продуктивності
            performance_metrics = self._calculate_performance_metrics()
            self.portfolio['performance_metrics'] = performance_metrics
            
            # Підготовка результатів
            results = {
                'status': 'success',
                'backtest_config': self.backtest_config,
                'tickers': self.config.data.tickers,
                'timeframes': self.config.data.timeframes,
                'strategies': list(strategies.keys()),
                'date_range': {
                    'start': date_range[0].isoformat() if date_range else None,
                    'end': date_range[-1].isoformat() if date_range else None
                },
                'portfolio': {
                    'initial_value': self.backtest_config['initial_capital'],
                    'final_value': performance_metrics.get('final_portfolio_value', 0),
                    'total_return': performance_metrics.get('total_return', 0),
                    'trades_count': len(self.portfolio['trades']),
                    'positions_count': len(self.portfolio['positions'])
                },
                'performance_metrics': performance_metrics,
                'equity_curve': self.portfolio['equity_curve'],
                'trades': self.portfolio['trades'],
                'data_sources': self.backtest_config['data_source_priority'],
                'performance_report': self.data_collector.get_performance_report()
            }
            
            # Збереження результатів
            results_file = self._save_backtest_results(results)
            results['results_file'] = results_file
            
            self.logger.info(f"Real data backtesting completed successfully")
            self.logger.info(f"Total return: {performance_metrics.get('total_return', 0):.2%}")
            self.logger.info(f"Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"Max drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Real data backtesting failed: {e}")
            return {
                'status': 'failed',
                'message': 'Real data backtesting process failed',
                'error': str(e)
            }
