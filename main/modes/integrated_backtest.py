#!/usr/bin/env python3
"""
Інтегрований покращений бектестинг
Об'єднує всі покращення: конфігурація, валідація, оптимізація, обробка помилок
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

from .base import BaseMode
from config.trading_config import TradingConfig
from utils.enhanced_data_validator import DataValidator
from utils.common_utils import (
    TechnicalIndicators, DataProcessor, CacheManager, 
    PerformanceMonitor, FileManager, ParallelProcessor,
    monitor_performance, handle_errors, retry_on_failure
)
from utils.enhanced_error_handler import (
    EnhancedErrorHandler, enhanced_error_handler, error_context
)


class IntegratedBacktestMode(BaseMode):
    """Інтегрований режим бектестингу з усіма покращеннями"""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        
        # Ініціалізація компонентів
        self.validator = DataValidator()
        self.data_processor = DataProcessor(config)
        self.technical_indicators = TechnicalIndicators()
        self.cache_manager = CacheManager(config.data.data_dir / "cache")
        self.performance_monitor = PerformanceMonitor()
        self.file_manager = FileManager(config.data.data_dir)
        self.parallel_processor = ParallelProcessor(max_workers=4)
        self.error_handler = EnhancedErrorHandler(self.logger)
        
        # Додавання callback для алертів
        self.error_handler.add_alert_callback(self._handle_alerts)
        
        self.logger.info("Integrated backtest mode initialized with all enhancements")
    
    def _handle_alerts(self, alert_type: str, alert_data: Dict[str, Any]):
        """Обробка алертів"""
        self.logger.warning(f"ALERT: {alert_type} - {alert_data}")
        
        # Збереження алертів в файл
        alert_file = self.file_manager.ensure_directory("alerts") / f"alert_{datetime.now().strftime('%Y%m%d')}.json"
        alerts = []
        if alert_file.exists():
            try:
                alerts = self.file_manager.load_json(alert_file)
            except:
                alerts = []
        
        alerts.append({
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'data': alert_data
        })
        
        self.file_manager.save_json(alerts, alert_file)
    
    @monitor_performance("load_historical_data")
    @retry_on_failure(max_retries=3, delay=1.0)
    def _load_historical_data_integrated(self) -> Dict[str, pd.DataFrame]:
        """Інтегроване завантаження історичних data"""
        historical_data = {}
        
        # Перевірка кешу
        cache_key = f"historical_data_{self.config.data.tickers}_{self.config.backtest.start_date}_{self.config.backtest.end_date}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            self.logger.info("Loaded historical data from cache")
            return cached_data
        
        # Генерація data
        dates = pd.date_range(
            start=self.config.backtest.start_date,
            end=self.config.backtest.end_date,
            freq='D'
        )
        
        # Векторизована генерація data
        n_dates = len(dates)
        n_tickers = len(self.config.data.tickers)
        
        # Генерація базових цін
        base_prices = np.random.uniform(50, 500, n_tickers)
        
        for i, ticker in enumerate(self.config.data.tickers):
            try:
                with error_context(f"load_data_{ticker}"):
                    # Валідація тікера
                    validated_ticker = self.validator.validate_trading_signal({
                        'ticker': ticker,
                        'action': 'HOLD',
                        'confidence': 0.0,
                        'timestamp': datetime.now()
                    }).ticker
                    
                    # Генерація data
                    np.random.seed(hash(ticker) % 1000)
                    price_base = base_prices[i]
                    
                    # Векторизована генерація OHLCV
                    returns = np.random.normal(0, 0.02, n_dates)
                    prices = price_base * (1 + np.cumsum(returns) * 0.001)
                    
                    # Генерація OHLC
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
                    
                    # Очищення та валідація data
                    data = self.data_processor.clean_ohlcv_data(data)
                    data = self.validator.validate_time_series_data(data, 'date')
                    
                    # Валідація OHLCV data
                    for _, row in data.iterrows():
                        self.validator.validate_ohlcv_data({
                            'ticker': validated_ticker,
                            'timestamp': row['date'],
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        })
                    
                    historical_data[validated_ticker] = data
                    
            except Exception as e:
                error_info = self.error_handler.handle_error(e, f"load_data_{ticker}")
                self.logger.error(f"Failed to load data for {ticker}: {e}")
                continue
        
        # Кешування результатів
        self.cache_manager.set(cache_key, historical_data, ttl=3600)
        
        self.logger.info(f"Loaded {len(historical_data)} tickers with {n_dates} data points each")
        return historical_data
    
    @monitor_performance("generate_signals")
    def _generate_signals_integrated(self, historical_data: Dict[str, pd.DataFrame], 
                                   strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Інтегрована генерація сигналів"""
        signals_data = {}
        
        # Паралельна обробка тікерів
        def process_ticker(ticker_data):
            ticker, data = ticker_data
            if len(data) < 100:
                return ticker, None
            
            try:
                with error_context(f"generate_signals_{ticker}_{strategy_name}"):
                    signals = self._generate_ticker_signals_integrated(data, strategy_name, strategy_config['parameters'])
                    return ticker, signals
            except Exception as e:
                self.error_handler.handle_error(e, f"generate_signals_{ticker}_{strategy_name}")
                return ticker, None
        
        # Паралельна обробка
        results = self.parallel_processor.process_dict_parallel(
            process_ticker, historical_data
        )
        
        # Фільтрація результатів
        for ticker, signals in results.items():
            if signals is not None and not signals.empty:
                signals_data[ticker] = signals
        
        return signals_data
    
    def _generate_ticker_signals_integrated(self, data: pd.DataFrame, strategy_name: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Інтегрована генерація сигналів для одного тікера"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 'HOLD'
        signals['confidence'] = 0.0
        
        try:
            if strategy_name == 'momentum':
                signals = self._momentum_signals_integrated(data, params, signals)
            elif strategy_name == 'mean_reversion':
                signals = self._mean_reversion_signals_integrated(data, params, signals)
            elif strategy_name == 'rsi_mean_reversion':
                signals = self._rsi_signals_integrated(data, params, signals)
            elif strategy_name == 'bollinger_bands':
                signals = self._bollinger_signals_integrated(data, params, signals)
            elif strategy_name == 'macd_crossover':
                signals = self._macd_signals_integrated(data, params, signals)
            # Додати інші стратегії...
            
        except Exception as e:
            self.error_handler.handle_error(e, f"generate_signals_{strategy_name}")
        
        return signals
    
    def _momentum_signals_integrated(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Інтегрована стратегія моментуму"""
        lookback = params['lookback_period']
        threshold = params['momentum_threshold']
        
        # Використання спільних утиліт
        returns = self.data_processor.calculate_returns(data['close'], lookback)
        
        # Валідація сигналів
        buy_signals = returns > threshold
        sell_signals = returns < -threshold
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.abs(returns[buy_signals | sell_signals])
        
        return signals
    
    def _mean_reversion_signals_integrated(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Інтегрована стратегія середнього повернення"""
        lookback = params['lookback_period']
        threshold = params['deviation_threshold']
        
        # Використання спільних утиліт
        rolling_mean = data['close'].rolling(lookback).mean()
        rolling_std = data['close'].rolling(lookback).std()
        z_score = (data['close'] - rolling_mean) / rolling_std
        
        buy_signals = z_score < -threshold
        sell_signals = z_score > threshold
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.abs(z_score[buy_signals | sell_signals]) / threshold
        
        return signals
    
    def _rsi_signals_integrated(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Інтегрована RSI стратегія"""
        period = params['rsi_period']
        oversold = params['oversold_threshold']
        overbought = params['overbought_threshold']
        
        # Використання спільних технічних індикаторів
        rsi = self.technical_indicators.calculate_rsi(data['close'], period)
        
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
    
    def _bollinger_signals_integrated(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Інтегрована Bollinger Bands стратегія"""
        period = params['bb_period']
        std = params['bb_std']
        
        # Використання спільних технічних індикаторів
        upper_band, rolling_mean, lower_band = self.technical_indicators.calculate_bollinger_bands(
            data['close'], period, std
        )
        
        price_shifted = data['close'].shift(1)
        upper_shifted = upper_band.shift(1)
        lower_shifted = lower_band.shift(1)
        
        buy_signals = (price_shifted <= upper_shifted) & (data['close'] > upper_band)
        sell_signals = (price_shifted >= lower_shifted) & (data['close'] < lower_band)
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        
        buy_confidence = (data['close'] - upper_band) / (upper_band - rolling_mean)
        sell_confidence = (lower_band - data['close']) / (rolling_mean - lower_band)
        
        signals.loc[buy_signals, 'confidence'] = np.abs(buy_confidence[buy_signals])
        signals.loc[sell_signals, 'confidence'] = np.abs(sell_confidence[sell_signals])
        
        return signals
    
    def _macd_signals_integrated(self, data: pd.DataFrame, params: Dict[str, Any], signals: pd.DataFrame) -> pd.DataFrame:
        """Інтегрована MACD стратегія"""
        fast = params['macd_fast']
        slow = params['macd_slow']
        signal = params['macd_signal']
        
        # Використання спільних технічних індикаторів
        macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(
            data['close'], fast, slow, signal
        )
        
        macd_shifted = macd_line.shift(1)
        signal_shifted = signal_line.shift(1)
        
        buy_signals = (macd_shifted <= signal_shifted) & (macd_line > signal_line)
        sell_signals = (macd_shifted >= signal_shifted) & (macd_line < signal_line)
        
        signals.loc[buy_signals, 'signal'] = 'BUY'
        signals.loc[sell_signals, 'signal'] = 'SELL'
        
        divergence = np.abs(macd_line - signal_line)
        signals.loc[buy_signals | sell_signals, 'confidence'] = np.minimum(
            divergence[buy_signals | sell_signals] / (np.std(divergence) + 1e-8), 1.0
        )
        
        return signals
    
    @monitor_performance("execute_trades")
    def _execute_trades_integrated(self, signals_data: Dict[str, pd.DataFrame], 
                                 historical_data: Dict[str, pd.DataFrame],
                                 strategy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Інтегроване виконання торгів"""
        trades = []
        position_size = strategy_config['parameters']['position_size']
        
        # Валідація та об'єднання сигналів
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
        
        # Сортування та обробка
        combined_signals = combined_signals.sort_index()
        
        for date, signal_group in combined_signals.groupby(level=0):
            date_trades = self._process_date_trades_integrated(date, signal_group, historical_data, position_size)
            trades.extend(date_trades)
        
        return trades
    
    def _process_date_trades_integrated(self, date: pd.Timestamp, signal_group: pd.DataFrame, 
                                       historical_data: Dict[str, pd.DataFrame], position_size: float) -> List[Dict[str, Any]]:
        """Інтегрована обробка торгів за датою"""
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
                with error_context(f"execute_trade_{ticker}_{action}"):
                    # Валідація торгової операції
                    trade_data = {
                        'date': date,
                        'ticker': ticker,
                        'action': action,
                        'price': current_price,
                        'confidence': confidence
                    }
                    
                    validated_trade = self.validator.validate_trade_order({
                        **trade_data,
                        'order_type': 'MARKET',
                        'quantity': int(self.portfolio['cash'] * position_size / current_price)
                    })
                    
                    # Виконання торгів
                    if action == 'BUY' and ticker not in self.portfolio['positions']:
                        trade = self._execute_buy_order_integrated(validated_trade, current_price)
                        if trade:
                            date_trades.append(trade)
                    
                    elif action == 'SELL' and ticker in self.portfolio['positions']:
                        trade = self._execute_sell_order_integrated(validated_trade, current_price)
                        if trade:
                            date_trades.append(trade)
                            
            except Exception as e:
                self.error_handler.handle_error(e, f"execute_trade_{ticker}_{action}")
                continue
        
        return date_trades
    
    def _execute_buy_order_integrated(self, order: Any, current_price: float) -> Optional[Dict[str, Any]]:
        """Інтегроване виконання ордера на купівлю"""
        shares_to_buy = order.quantity
        cost = shares_to_buy * current_price * (1 + self.config.backtest.commission)
        
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
                'confidence': order.confidence,
                'commission': cost - (shares_to_buy * current_price)
            }
        
        return None
    
    def _execute_sell_order_integrated(self, order: Any, current_price: float) -> Optional[Dict[str, Any]]:
        """Інтегроване виконання ордера на продаж"""
        if order.ticker not in self.portfolio['positions']:
            return None
        
        position = self.portfolio['positions'][order.ticker]
        shares_to_sell = position['shares']
        
        if shares_to_sell > 0:
            revenue = shares_to_sell * current_price * (1 - self.config.backtest.commission)
            
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
                'confidence': order.confidence,
                'commission': (shares_to_sell * current_price) - revenue
            }
        
        return None
    
    @monitor_performance("calculate_performance")
    def _calculate_performance_metrics_integrated(self, portfolio_values: List[float], 
                                                trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Інтегрований розрахунок метрик продуктивності"""
        if not portfolio_values:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        
        # Базові метрики
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Доходності
        daily_returns = self.data_processor.calculate_returns(pd.Series(portfolio_values))
        
        # Метрики
        metrics = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'daily_return_mean': daily_returns.mean(),
            'daily_return_std': daily_returns.std(),
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
                'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
                'total_commission': sum(t.get('commission', 0) for t in trades)
            })
        
        return metrics
    
    @monitor_performance("run_parallel_strategies")
    def run_parallel_strategies_integrated(self, historical_data: Dict[str, pd.DataFrame], 
                                         strategies: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Інтегроване паралельне виконання стратегій"""
        results = {}
        
        # Валідація запиту бектестингу
        backtest_request = {
            'tickers': list(historical_data.keys()),
            'timeframes': self.config.data.timeframes,
            'start_date': datetime.strptime(self.config.backtest.start_date, '%Y-%m-%d').date(),
            'end_date': datetime.strptime(self.config.backtest.end_date, '%Y-%m-%d').date(),
            'initial_capital': self.config.risk.initial_capital,
            'strategies': list(strategies.keys())
        }
        
        try:
            validated_request = self.validator.validate_backtest_request(backtest_request)
        except Exception as e:
            self.error_handler.handle_error(e, "validate_backtest_request")
            return {'error': 'Invalid backtest request'}
        
        # Паралельне виконання
        with ThreadPoolExecutor(max_workers=min(4, len(strategies))) as executor:
            future_to_strategy = {
                executor.submit(self._run_single_strategy_integrated, historical_data, strategy_name, strategy_config): strategy_name
                for strategy_name, strategy_config in strategies.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy_name] = result
                    return_pct = result.get('metrics', {}).get('total_return', 0)
                    self.logger.info(f"Completed {strategy_name}: {return_pct:.2%} return")
                except Exception as e:
                    error_info = self.error_handler.handle_error(e, f"run_strategy_{strategy_name}")
                    results[strategy_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'error_info': error_info.to_dict()
                    }
        
        return results
    
    def _run_single_strategy_integrated(self, historical_data: Dict[str, pd.DataFrame], 
                                       strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Інтегроване виконання однієї стратегії"""
        try:
            with error_context(f"run_strategy_{strategy_name}"):
                # Ініціалізація портфеля
                self.portfolio = {
                    'cash': self.config.risk.initial_capital,
                    'positions': {},
                    'transactions': []
                }
                
                # Генерація сигналів
                signals_data = self._generate_signals_integrated(historical_data, strategy_name, strategy_config)
                
                # Виконання торгів
                trades = self._execute_trades_integrated(signals_data, historical_data, strategy_config)
                
                # Розрахунок метрик
                portfolio_values = self._calculate_portfolio_values_integrated(historical_data, trades)
                metrics = self._calculate_performance_metrics_integrated(portfolio_values, trades)
                
                return {
                    'status': 'success',
                    'strategy': strategy_name,
                    'metrics': metrics,
                    'trades': trades,
                    'portfolio_values': portfolio_values,
                    'signals_count': sum(len(signals) for signals in signals_data.values())
                }
                
        except Exception as e:
            error_info = self.error_handler.handle_error(e, f"run_strategy_{strategy_name}")
            return {
                'status': 'failed',
                'strategy': strategy_name,
                'error': str(e),
                'error_info': error_info.to_dict()
            }
    
    def _calculate_portfolio_values_integrated(self, historical_data: Dict[str, pd.DataFrame], 
                                            trades: List[Dict[str, Any]]) -> List[float]:
        """Інтегрований розрахунок значень портфеля"""
        if not trades:
            return [self.config.risk.initial_capital]
        
        # Створення DataFrame з торгами
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df = trades_df.sort_values('date')
        
        # Отримання всіх дат
        all_dates = set()
        for ticker, data in historical_data.items():
            all_dates.update(data.index)
        
        all_dates = sorted(all_dates)
        
        # Розрахунок значень портфеля
        portfolio_values = []
        current_positions = {}
        current_cash = self.config.risk.initial_capital
        
        for date in all_dates:
            # Оновлення позицій
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
            
            # Розрахунок вартості позицій
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
        """Запуск інтегрованого бектестингу"""
        self.logger.info("Starting integrated backtesting with all enhancements...")
        
        try:
            with error_context("integrated_backtest_run"):
                # Перевірка передумов
                if not self.validate_prerequisites():
                    return {
                        'status': 'failed',
                        'error': 'Prerequisites validation failed',
                        'message': 'Cannot start backtesting - missing requirements'
                    }
                
                # Завантаження data
                self.logger.info("Loading historical data...")
                historical_data = self._load_historical_data_integrated()
                
                # Ініціалізація стратегій
                self.logger.info("Initializing trading strategies...")
                strategies = self._initialize_strategies()
                
                # Паралельне виконання стратегій
                self.logger.info("Running parallel backtest simulations...")
                backtest_results = self.run_parallel_strategies_integrated(historical_data, strategies)
                
                # Аналіз результатів
                self.logger.info("Analyzing backtest results...")
                analysis_results = self._analyze_backtest_results(backtest_results)
                
                # Генерація звітів
                self.logger.info("Generating comprehensive reports...")
                reports = self._generate_reports(backtest_results, analysis_results)
                
                # Збереження результатів
                results = self._save_backtest_results(backtest_results, analysis_results, reports)
                
                # Додавання статистики продуктивності
                results['performance_stats'] = self.performance_monitor.get_performance_report()
                results['error_stats'] = self.error_handler.get_error_statistics()
                
                self.logger.info("Integrated backtesting completed successfully")
                return results
                
        except Exception as e:
            error_info = self.error_handler.handle_error(e, "integrated_backtest_run")
            return {
                'status': 'failed',
                'error': str(e),
                'error_info': error_info.to_dict(),
                'message': 'Integrated backtesting process failed'
            }
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очищення ресурсів"""
        try:
            # Очищення кешу
            self.cache_manager.clear()
            
            # Експорт статистики
            stats_file = self.file_manager.ensure_directory("stats") / f"backtest_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            stats = {
                'performance': self.performance_monitor.get_performance_report(),
                'errors': self.error_handler.get_error_statistics()
            }
            self.file_manager.save_json(stats, stats_file)
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
