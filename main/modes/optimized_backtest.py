#!/usr/bin/env python3
"""
Оптимізований бектестинг з векторизованими операціями
Покращена продуктивність через використання pandas/numpy оптимізацій
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

from .base import BaseMode
from config.trading_config import TradingConfig
from utils.enhanced_data_validator import DataValidator
from utils.common_utils import CacheManager, PerformanceMonitor
from utils.enhanced_error_handler import EnhancedErrorHandler


class OptimizedBacktestMode(BaseMode):
    """Оптимізований режим бектестингу з векторизованими операціями"""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.validator = DataValidator()
        self.cache_manager = CacheManager()
        self.performance_tracker = PerformanceMonitor()
        self.error_handler = EnhancedErrorHandler()
        
        # Ініціалізація конфігурації бектестингу
        self.backtest_config = {
            'start_date': datetime.now() - timedelta(days=365),
            'end_date': datetime.now(),
            'commission': 0.001,
            'slippage': 0.0001,
            'initial_capital': self.config.risk.initial_capital
        }
        
    def _load_historical_data_optimized(self) -> Dict[str, pd.DataFrame]:
        """Оптимізоване завантаження історичних data з кешуванням"""
        historical_data = {}
        
        # Генерація data для всіх тікерів одночасно
        dates = pd.date_range(
            start=self.backtest_config['start_date'],
            end=self.backtest_config['end_date'],
            freq='D'
        )
        
        # Векторизована генерація data
        n_dates = len(dates)
        n_tickers = len(self.config.data.tickers)
        
        # Генерація базових цін для всіх тікерів
        base_prices = np.random.uniform(50, 500, n_tickers)
        
        for i, ticker in enumerate(self.config.data.tickers):
            try:
                # Використання кешу
                cache_key = f"{ticker}_{self.backtest_config['start_date']}_{self.backtest_config['end_date']}"
                if cache_key in self.data_cache:
                    historical_data[ticker] = self.data_cache[cache_key]
                    continue
                
                # Векторизована генерація OHLCV
                np.random.seed(hash(ticker) % 1000)
                price_base = base_prices[i]
                
                # Генерація змін цин векторизовано
                returns = np.random.normal(0, 0.02, n_dates)
                prices = price_base * (1 + np.cumsum(returns) * 0.001)
                
                # Векторизована генерація OHLC
                high_noise = np.random.uniform(0, 0.02, n_dates)
                low_noise = np.random.uniform(0, 0.02, n_dates)
                
                data = pd.DataFrame({
                    'date': dates,
                    'open': prices * (1 - low_noise * 0.5),
                    'high': prices * (1 + high_noise),
                    'low': prices * (1 - low_noise),
                    'close': prices,
                    'volume': np.random.randint(100000, 10000000, n_dates)
                })
                
                # Виправлення OHLC логіки векторизовано
                data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
                data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
                
                # Валідація data
                data = self.validator.validate_time_series_data(data, 'date')
                
                historical_data[ticker] = data
                self.data_cache[cache_key] = data
                
                self.logger.info(f"Loaded {len(data)} data points for {ticker}")
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {ticker}: {e}")
                continue
        
        return historical_data
    
    @lru_cache(maxsize=128)
    def _calculate_technical_indicators(self, ticker: str, data_hash: int) -> Dict[str, np.ndarray]:
        """Кешування розрахунків технічних індикаторів"""
        # Цей метод буде перевизначений в підкласах
        return {}
    
    def _generate_signals_vectorized(self, historical_data: Dict[str, pd.DataFrame], 
                                   strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Векторизована генерація сигналів для всіх тікерів"""
        signals_data = {}
        
        for ticker, data in historical_data.items():
            if len(data) < 100:
                continue
            
            # Векторизована генерація сигналів
            signals = self._generate_ticker_signals_vectorized(data, strategy_name, strategy_config['parameters'])
            
            if not signals.empty:
                signals_data[ticker] = signals
        
        return signals_data
    
    def _generate_ticker_signals_vectorized(self, data: pd.DataFrame, strategy_name: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Векторизована генерація сигналів для одного тікера"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 'HOLD'
        signals['confidence'] = 0.0
        
        try:
            if strategy_name == 'momentum':
                signals = self._momentum_signals_vectorized(data, params, signals)
            elif strategy_name == 'mean_reversion':
                signals = self._mean_reversion_signals_vectorized(data, params, signals)
            elif strategy_name == 'rsi_mean_reversion':
                signals = self._rsi_signals_vectorized(data, params, signals)
            elif strategy_name == 'bollinger_bands':
                signals = self._bollinger_signals_vectorized(data, params, signals)
            elif strategy_name == 'macd_crossover':
                signals = self._macd_signals_vectorized(data, params, signals)
            # Додати інші стратегії...
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {strategy_name}: {e}")
        
        return signals
    
    def _momentum_signals_vectorized(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Векторизована стратегія моментуму"""
        lookback = params['lookback_period']
        threshold = params['momentum_threshold']
        
        # Векторизований розрахунок моментуму
        returns = data['close'].pct_change(lookback)
        
        # Векторизована генерація сигналів
        buy_signals = returns > threshold
        sell_signals = returns < -threshold
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.abs(returns[buy_signals | sell_signals])
        
        return signals
    
    def _mean_reversion_signals_vectorized(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Векторизована стратегія середнього повернення"""
        lookback = params['lookback_period']
        threshold = params['deviation_threshold']
        
        # Векторизований розрахунок середнього та стандартного відхилення
        rolling_mean = data['close'].rolling(lookback).mean()
        rolling_std = data['close'].rolling(lookback).std()
        z_score = (data['close'] - rolling_mean) / rolling_std
        
        # Векторизована генерація сигналів
        buy_signals = z_score < -threshold
        sell_signals = z_score > threshold
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.abs(z_score[buy_signals | sell_signals]) / threshold
        
        return signals
    
    def _rsi_signals_vectorized(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Векторизована RSI стратегія"""
        period = params['rsi_period']
        oversold = params['oversold_threshold']
        overbought = params['overbought_threshold']
        
        # Векторизований розрахунок RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Векторизована генерація сигналів
        rsi_shifted = rsi.shift(1)
        buy_signals = (rsi_shifted <= oversold) & (rsi > oversold)
        sell_signals = (rsi_shifted >= overbought) & (rsi < overbought)
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.where(
            buy_signals, (overbought - rsi[buy_signals]) / (overbought - oversold),
            (rsi[sell_signals] - oversold) / (overbought - oversold)
        )
        
        return signals
    
    def _bollinger_signals_vectorized(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Векторизована Bollinger Bands стратегія"""
        period = params['bb_period']
        std = params['bb_std']
        
        # Векторизований розрахунок Bollinger Bands
        rolling_mean = data['close'].rolling(period).mean()
        rolling_std = data['close'].rolling(period).std()
        upper_band = rolling_mean + (rolling_std * std)
        lower_band = rolling_mean - (rolling_std * std)
        
        # Векторизована генерація сигналів
        price_shifted = data['close'].shift(1)
        upper_shifted = upper_band.shift(1)
        lower_shifted = lower_band.shift(1)
        
        buy_signals = (price_shifted <= upper_shifted) & (data['close'] > upper_band)
        sell_signals = (price_shifted >= lower_shifted) & (data['close'] < lower_band)
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        
        # Розрахунок впевненості на основі відстані від смуг
        buy_confidence = (data['close'] - upper_band) / (upper_band - rolling_mean)
        sell_confidence = (lower_band - data['close']) / (rolling_mean - lower_band)
        
        signals.loc[buy_signals, 'confidence'] = np.abs(buy_confidence[buy_signals])
        signals.loc[sell_signals, 'confidence'] = np.abs(sell_confidence[sell_signals])
        
        return signals
    
    def _macd_signals_vectorized(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Векторизована MACD стратегія"""
        fast = params['macd_fast']
        slow = params['macd_slow']
        signal = params['macd_signal']
        
        # Векторизований розрахунок MACD
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Векторизована генерація сигналів
        macd_shifted = macd_line.shift(1)
        signal_shifted = signal_line.shift(1)
        
        buy_signals = (macd_shifted <= signal_shifted) & (macd_line > signal_line)
        sell_signals = (macd_shifted >= signal_shifted) & (macd_line < signal_line)
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        
        # Розрахунок впевненості на основі відстані між лініями
        divergence = np.abs(macd_line - signal_line)
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.minimum(
            divergence[buy_signals | sell_signals] / (np.std(divergence) + 1e-8), 1.0
        )
        
        return signals
    
    def _execute_trades_vectorized(self, signals_data: Dict[str, pd.DataFrame], 
                                 historical_data: Dict[str, pd.DataFrame],
                                 strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Векторизоване виконання торгів"""
        trades = []
        position_size = strategy_config['parameters']['position_size']
        
        # Об'єднання всіх сигналів в один DataFrame
        all_signals = []
        for ticker, signals in signals_data.items():
            ticker_signals = signals.copy()
            ticker_signals['ticker'] = ticker
            all_signals.append(ticker_signals)
        
        if not all_signals:
            return trades
        
        combined_signals = pd.concat(all_signals, ignore_index=True)
        combined_signals = combined_signals[combined_signals['signal'] != 'HOLD']
        
        if combined_signals.empty:
            return trades
        
        # Сортування по даті
        combined_signals = combined_signals.sort_index()
        
        # Векторизована обробка торгів
        for date, signal_group in combined_signals.groupby(level=0):
            date_trades = self._process_date_trades(date, signal_group, historical_data, position_size)
            trades.extend(date_trades)
        
        return trades
    
    def _process_date_trades(self, date: pd.Timestamp, signal_group: pd.DataFrame, 
                           historical_data: Dict[str, pd.DataFrame], position_size: float) -> List[Dict[str, Any]]:
        """Обробка торгів за конкретну дату"""
        date_trades = []
        
        for _, signal in signal_group.iterrows():
            ticker = signal['ticker']
            action = signal['signal']
            confidence = signal['confidence']
            
            if ticker not in historical_data:
                continue
            
            ticker_data = historical_data[ticker]
            current_price_data = ticker_data[ticker_data.index == date]
            
            if current_price_data.empty:
                continue
            
            current_price = current_price_data['close'].iloc[0]
            
            try:
                validated_trade = self.validator.validate_trade_order({
                    **trade_data,
                    'order_type': 'MARKET',
                    'quantity': int(self.portfolio['cash'] * position_size / current_price)
                })
                
                # Виконання торгів
                if action == 'BUY' and ticker not in self.portfolio['positions']:
                    trade = self._execute_buy_order(validated_trade, current_price)
                    if trade:
                        date_trades.append(trade)
                
                elif action == 'SELL' and ticker in self.portfolio['positions']:
                    trade = self._execute_sell_order(validated_trade, current_price)
                    if trade:
                        date_trades.append(trade)
                        
            except ValueError as e:
                self.logger.warning(f"Invalid trade for {ticker}: {e}")
                continue
        
        return date_trades
    
    def _execute_buy_order(self, order: Any, current_price: float) -> Optional[Dict[str, Any]]:
        """Виконання ордера на купівлю"""
        shares_to_buy = order.quantity
        cost = shares_to_buy * current_price * (1 + self.backtest_config['commission'])
        
        if cost <= self.portfolio['cash']:
            self.portfolio['cash'] -= cost
            self.portfolio['positions'][order.ticker] = {
                'shares': shares_to_buy,
                'entry_price': current_price,
                'entry_date': order.date
            }
            
            return {
                'date': order.date,
                'ticker': order.ticker,
                'action': 'BUY',
                'shares': shares_to_buy,
                'price': current_price,
                'cost': cost,
                'confidence': order.confidence
            }
        
        return None
    
    def _execute_sell_order(self, order: Any, current_price: float) -> Optional[Dict[str, Any]]:
        """Виконання ордера на продаж"""
        if order.ticker not in self.portfolio['positions']:
            return None
        
        position = self.portfolio['positions'][order.ticker]
        shares_to_sell = position['shares']
        
        if shares_to_sell > 0:
            revenue = shares_to_sell * current_price * (1 - self.backtest_config['commission'])
            
            self.portfolio['cash'] += revenue
            del self.portfolio['positions'][order.ticker]
            
            return {
                'date': order.date,
                'ticker': order.ticker,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'revenue': revenue,
                'profit': revenue - (shares_to_sell * position['entry_price']),
                'confidence': order.confidence
            }
        
        return None
    
    def _calculate_performance_metrics_optimized(self, portfolio_values: List[float], 
                                              trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Оптимізований розрахунок метрик продуктивності"""
        if not portfolio_values:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        
        # Векторизовані розрахунки
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Векторизований розрахунок доходностей
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Розрахунок метрик
        metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'daily_return_mean': np.mean(daily_returns) if len(daily_returns) > 0 else 0,
            'daily_return_std': np.std(daily_returns) if len(daily_returns) > 0 else 0,
        }
        
        # Sharpe ratio
        if metrics['daily_return_std'] > 0:
            metrics['sharpe_ratio'] = metrics['daily_return_mean'] / metrics['daily_return_std'] * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Волатильність
        metrics['volatility'] = metrics['daily_return_std'] * np.sqrt(252)
        
        # Аналіз торгів
        if trades:
            profits = [t.get('profit', 0) for t in trades if 'profit' in t]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            metrics.update({
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(profits) if profits else 0,
                'average_profit_per_trade': np.mean(profits) if profits else 0,
                'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
            })
        
        return metrics
    
    def run_parallel_strategies(self, historical_data: Dict[str, pd.DataFrame], 
                              strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Паралельне виконання стратегій"""
        results = {}
        
        # Використання ThreadPoolExecutor для паралельної обробки
        with ThreadPoolExecutor(max_workers=min(4, len(strategies))) as executor:
            future_to_strategy = {
                executor.submit(self._run_single_strategy_optimized, historical_data, strategy_name, strategy_config): strategy_name
                for strategy_name, strategy_config in strategies.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy_name] = result
                    self.logger.info(f"Completed {strategy_name}: {result.get('metrics', {}).get('total_return', 0):.2%} return")
                except Exception as e:
                    self.logger.error(f"Failed to run {strategy_name}: {e}")
                    results[strategy_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return results
    
    def _run_single_strategy_optimized(self, historical_data: Dict[str, pd.DataFrame], 
                                     strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимізоване виконання однієї стратегії"""
        try:
            # Ініціалізація портфеля
            self.portfolio = {
                'cash': self.backtest_config['initial_capital'],
                'positions': {},
                'transactions': []
            }
            
            # Векторизована генерація сигналів
            signals_data = self._generate_signals_vectorized(historical_data, strategy_name, strategy_config)
            
            # Векторизоване виконання торгів
            trades = self._execute_trades_vectorized(signals_data, historical_data, strategy_config)
            
            # Розрахунок метрик
            portfolio_values = self._calculate_portfolio_values(historical_data, trades)
            metrics = self._calculate_performance_metrics_optimized(portfolio_values, trades)
            
            return {
                'status': 'success',
                'strategy': strategy_name,
                'metrics': metrics,
                'trades': trades,
                'portfolio_values': portfolio_values
            }
            
        except Exception as e:
            self.logger.error(f"Strategy {strategy_name} failed: {e}")
            return {
                'status': 'failed',
                'strategy': strategy_name,
                'error': str(e)
            }
    
    def _calculate_portfolio_values(self, historical_data: Dict[str, pd.DataFrame], 
                                  trades: List[Dict[str, Any]]) -> List[float]:
        """Розрахунок значень портфеля в часі"""
        if not trades:
            return [self.backtest_config['initial_capital']]
        
        # Створення DataFrame з торгами
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df = trades_df.sort_values('date')
        
        # Отримання всіх унікальних дат
        all_dates = set()
        for ticker, data in historical_data.items():
            all_dates.update(data.index)
        
        all_dates = sorted(all_dates)
        
        # Розрахунок значень портфеля для кожної дати
        portfolio_values = []
        current_positions = {}
        current_cash = self.backtest_config['initial_capital']
        
        for date in all_dates:
            # Оновлення позицій на основі торгів
            date_trades = trades_df[trades_df['date'] == date]
            
            for _, trade in date_trades.iterrows():
                if trade['action'] == 'BUY':
                    current_positions[trade['ticker']] = current_positions.get(trade['ticker'], 0) + trade['shares']
                    current_cash -= trade['cost']
                elif trade['action'] == 'SELL':
                    current_positions[trade['ticker']] = current_positions.get(trade['ticker'], 0) - trade['shares']
                    if current_positions[trade['ticker']] <= 0:
                        del current_positions[trade['ticker']]
                    current_cash += trade['revenue']
            
            # Розрахунок поточної вартості позицій
            positions_value = 0
            for ticker, shares in current_positions.items():
                if ticker in historical_data:
                    ticker_data = historical_data[ticker]
                    if date in ticker_data.index:
                        current_price = ticker_data.loc[date, 'close']
                        positions_value += shares * current_price
            
            total_value = current_cash + positions_value
            portfolio_values.append(total_value)
        
        return portfolio_values
    
    def run(self) -> Dict[str, Any]:
        """Запуск оптимізованого бектестингу"""
        self.logger.info("Starting optimized backtesting...")
        
        try:
            # Перевірка передумов
            if not self.validate_prerequisites():
                return {
                    'status': 'failed',
                    'error': 'Prerequisites validation failed',
                    'message': 'Cannot start backtesting - missing requirements'
                }
            
            # Оптимізоване завантаження data
            self.logger.info("Loading historical data...")
            historical_data = self._load_historical_data_optimized()
            
            # Ініціалізація стратегій
            self.logger.info("Initializing trading strategies...")
            strategies = self._initialize_strategies()
            
            # Паралельне виконання стратегій
            self.logger.info("Running parallel backtest simulations...")
            backtest_results = self.run_parallel_strategies(historical_data, strategies)
            
            # Аналіз результатів
            self.logger.info("Analyzing backtest results...")
            analysis_results = self._analyze_backtest_results(backtest_results)
            
            # Генерація звітів
            self.logger.info("Generating comprehensive reports...")
            reports = self._generate_reports(backtest_results, analysis_results)
            
            # Збереження результатів
            results = self._save_backtest_results(backtest_results, analysis_results, reports)
            
            self.logger.info("Optimized backtesting completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Optimized backtesting failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Optimized backtesting process failed'
            }
        finally:
            self.cleanup()
