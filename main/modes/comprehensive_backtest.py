#!/usr/bin/env python3
"""
Comprehensive Backtest Mode - розширений бектестинг з додатковими стратегіями
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using simplified implementations.")

from .backtest import BacktestMode
from config.trading_config import TradingConfig


class ComprehensiveBacktestMode(BacktestMode):
    """Розширений режим бектестингу з додатковими стратегіями та тікерами"""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.extended_tickers = self._get_extended_ticker_universe()
        self.extended_timeframes = ['1d', '1h', '15m', '5m']
    
    def _get_extended_ticker_universe(self) -> List[str]:
        """Розширений список тікерів для різних секторів ринку"""
        return {
            'tech_mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'tech_growth': ['AMD', 'PLTR', 'ROKU', 'SNAP', 'SPOT', 'ZM', 'CRM'],
            'semiconductors': ['NVDA', 'AMD', 'INTC', 'MU', 'SOXX', 'SMH', 'TSM'],
            'crypto_related': ['COIN', 'MARA', 'RIOT', 'SQ', 'PYPL', 'MSTR'],
            'energy': ['XOM', 'CVX', 'CLF', 'HAL', 'SLB', 'COP', 'EOG'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'V'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TSLA'],
            'consumer': ['AMZN', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX'],
            'industrial': ['CAT', 'DE', 'GE', 'BA', 'HON', 'UPS', 'MMM'],
            'etfs': ['QQQ', 'SPY', 'IWM', 'GLD', 'TLT', 'XLE', 'XLK', 'ARKK', 'SOXX']
        }
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Ініціалізація розширених торгових стратегій"""
        base_strategies = super()._initialize_strategies()
        
        # Додаткові просунуті стратегії
        advanced_strategies = {
            'rsi_mean_reversion': {
                'name': 'RSI Mean Reversion',
                'description': 'RSI-based mean reversion with dynamic thresholds',
                'parameters': {
                    'rsi_period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70,
                    'position_size': 0.12
                }
            },
            'bollinger_bands': {
                'name': 'Bollinger Bands Breakout',
                'description': 'Trade based on Bollinger Bands price action',
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'position_size': 0.10
                }
            },
            'macd_crossover': {
                'name': 'MACD Crossover',
                'description': 'MACD line crossover strategy',
                'parameters': {
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'position_size': 0.11
                }
            },
            'volume_price': {
                'name': 'Volume Price Analysis',
                'description': 'Volume-weighted price action strategy',
                'parameters': {
                    'volume_ma_period': 20,
                    'volume_threshold': 1.5,
                    'position_size': 0.09
                }
            },
            'atr_breakout': {
                'name': 'ATR Breakout',
                'description': 'Average True Range breakout strategy',
                'parameters': {
                    'atr_period': 14,
                    'atr_multiplier': 2.0,
                    'position_size': 0.08
                }
            },
            'stochastic_oscillator': {
                'name': 'Stochastic Oscillator',
                'description': 'Stochastic-based momentum strategy',
                'parameters': {
                    'stoch_k': 14,
                    'stoch_d': 3,
                    'overbought': 80,
                    'oversold': 20,
                    'position_size': 0.10
                }
            },
            'williams_r': {
                'name': 'Williams %R',
                'description': 'Williams %R reversal strategy',
                'parameters': {
                    'williams_period': 14,
                    'overbought': -20,
                    'oversold': -80,
                    'position_size': 0.09
                }
            },
            'cci_strategy': {
                'name': 'Commodity Channel Index',
                'description': 'CCI-based trend following strategy',
                'parameters': {
                    'cci_period': 20,
                    'cci_threshold': 100,
                    'position_size': 0.10
                }
            },
            'multi_timeframe': {
                'name': 'Multi-Timeframe Analysis',
                'description': 'Combine signals from multiple timeframes',
                'parameters': {
                    'primary_timeframe': '1d',
                    'secondary_timeframe': '1h',
                    'signal_alignment_threshold': 0.7,
                    'position_size': 0.12
                }
            },
            'sector_rotation': {
                'name': 'Sector Rotation',
                'description': 'Rotate between sectors based on relative strength',
                'parameters': {
                    'lookback_period': 60,
                    'sector_momentum_threshold': 0.05,
                    'position_size': 0.15
                }
            },
            'pairs_trading': {
                'name': 'Pairs Trading',
                'description': 'Statistical arbitrage between correlated assets',
                'parameters': {
                    'correlation_threshold': 0.7,
                    'zscore_entry': 2.0,
                    'zscore_exit': 0.5,
                    'position_size': 0.10
                }
            },
            'volatility_mean_reversion': {
                'name': 'Volatility Mean Reversion',
                'description': 'Trade volatility contractions/expansions',
                'parameters': {
                    'volatility_window': 20,
                    'volatility_percentile': 20,
                    'position_size': 0.08
                }
            }
        }
        
        # Об'єднання базових та просунутих стратегій
        all_strategies = {**base_strategies, **advanced_strategies}
        
        self.logger.info(f"Initialized {len(all_strategies)} comprehensive trading strategies")
        return all_strategies
    
    def _generate_signals(self, date: datetime, historical_data: Dict[str, pd.DataFrame], 
                         strategy_name: str, strategy_config: Dict[str, Any]) -> Dict[str, str]:
        """Генерація торгових сигналів для розширених стратегій"""
        signals = {}
        
        for ticker in self.config.data.tickers:
            if ticker not in historical_data:
                continue
            
            ticker_data = historical_data[ticker]
            current_data = ticker_data[ticker_data['date'] <= date]
            
            if len(current_data) < 100:  # Збільшено мінімальну кількість data
                continue
            
            # Генерація сигналів залежно від стратегії
            if strategy_name == 'rsi_mean_reversion':
                signal = self._generate_rsi_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'bollinger_bands':
                signal = self._generate_bollinger_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'macd_crossover':
                signal = self._generate_macd_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'volume_price':
                signal = self._generate_volume_price_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'atr_breakout':
                signal = self._generate_atr_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'stochastic_oscillator':
                signal = self._generate_stochastic_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'williams_r':
                signal = self._generate_williams_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'cci_strategy':
                signal = self._generate_cci_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'multi_timeframe':
                signal = self._generate_multi_timeframe_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'sector_rotation':
                signal = self._generate_sector_rotation_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'pairs_trading':
                signal = self._generate_pairs_signal(current_data, strategy_config['parameters'])
            elif strategy_name == 'volatility_mean_reversion':
                signal = self._generate_volatility_mean_reversion_signal(current_data, strategy_config['parameters'])
            else:
                # Використовуємо базові стратегії з батьківського класу
                signal = super()._generate_signals(date, {ticker: ticker_data}, strategy_name, strategy_config).get(ticker, 'HOLD')
            
            signals[ticker] = signal
        
        return signals
    
    def _generate_rsi_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """RSI mean reversion стратегія"""
        period = params['rsi_period']
        oversold = params['oversold_threshold']
        overbought = params['overbought_threshold']
        
        if len(data) < period:
            return 'HOLD'
        
        closes = data['close'].values
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(closes, timeperiod=period)
        else:
            # Спрощена реалізація RSI
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = np.array([100 - (100 / (1 + rs))] * len(closes))
        
        if len(rsi) < 2:
            return 'HOLD'
        
        current_rsi = rsi[-1]
        previous_rsi = rsi[-2]
        
        # Переворот з oversold
        if previous_rsi <= oversold and current_rsi > oversold:
            return 'BUY'
        # Переворот з overbought
        elif previous_rsi >= overbought and current_rsi < overbought:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_bollinger_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Bollinger Bands стратегія"""
        period = params['bb_period']
        std = params['bb_std']
        
        if len(data) < period:
            return 'HOLD'
        
        closes = data['close'].values
        
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(closes, timeperiod=period, nbdevup=std, nbdevdn=std)
        else:
            # Спрощена реалізація Bollinger Bands
            middle = pd.Series(closes).rolling(window=period).mean().values
            std_dev = pd.Series(closes).rolling(window=period).std().values
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
        
        if len(upper) < 2:
            return 'HOLD'
        
        current_price = closes[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        previous_price = closes[-2]
        previous_upper = upper[-2]
        previous_lower = lower[-2]
        
        # Пробій верхньої смуги
        if previous_price <= previous_upper and current_price > current_upper:
            return 'BUY'
        # Пробій нижньої смуги
        elif previous_price >= previous_lower and current_price < current_lower:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_macd_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """MACD crossover стратегія"""
        fast = params['macd_fast']
        slow = params['macd_slow']
        signal = params['macd_signal']
        
        if len(data) < slow:
            return 'HOLD'
        
        closes = data['close'].values
        
        if TALIB_AVAILABLE:
            macd, signal_line, histogram = talib.MACD(closes, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        else:
            # Спрощена реалізація MACD
            ema_fast = pd.Series(closes).ewm(span=fast).mean().values
            ema_slow = pd.Series(closes).ewm(span=slow).mean().values
            macd = ema_fast - ema_slow
            signal_line = pd.Series(macd).ewm(span=signal).mean().values
            histogram = macd - signal_line
        
        if len(macd) < 2:
            return 'HOLD'
        
        # MACD crossover signal line
        if macd[-2] <= signal_line[-2] and macd[-1] > signal_line[-1]:
            return 'BUY'
        elif macd[-2] >= signal_line[-2] and macd[-1] < signal_line[-1]:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_volume_price_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Volume Price стратегія"""
        volume_period = params['volume_ma_period']
        threshold = params['volume_threshold']
        
        if len(data) < volume_period:
            return 'HOLD'
        
        closes = data['close'].values
        volumes = data['volume'].values
        
        # Volume moving average
        if TALIB_AVAILABLE:
            volume_ma = talib.SMA(volumes, timeperiod=volume_period)
        else:
            volume_ma = pd.Series(volumes).rolling(window=volume_period).mean().values
        
        if len(volume_ma) < 2:
            return 'HOLD'
        
        current_volume = volumes[-1]
        avg_volume = volume_ma[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Price change
        price_change = (closes[-1] - closes[-2]) / closes[-2]
        
        # High volume + price movement
        if volume_ratio > threshold and price_change > 0.02:
            return 'BUY'
        elif volume_ratio > threshold and price_change < -0.02:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_atr_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """ATR breakout стратегія"""
        period = params['atr_period']
        multiplier = params['atr_multiplier']
        
        if len(data) < period:
            return 'HOLD'
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if TALIB_AVAILABLE:
            atr = talib.ATR(high, low, close, timeperiod=period)
        else:
            # Спрощена реалізація ATR
            high_low = high - low
            high_close = np.abs(high[:-1] - close[1:])
            low_close = np.abs(low[:-1] - close[1:])
            true_range = np.concatenate([[high_low[0]], np.maximum(high_low[1:], np.maximum(high_close, low_close))])
            atr = pd.Series(true_range).rolling(window=period).mean().values
        
        if len(atr) < 2:
            return 'HOLD'
        
        current_close = close[-1]
        previous_close = close[-2]
        current_atr = atr[-1]
        
        # ATR breakout
        upper_breakout = previous_close + (current_atr * multiplier)
        lower_breakout = previous_close - (current_atr * multiplier)
        
        if current_close > upper_breakout:
            return 'BUY'
        elif current_close < lower_breakout:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_stochastic_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Stochastic oscillator стратегія"""
        k_period = params['stoch_k']
        d_period = params['stoch_d']
        overbought = params['overbought']
        oversold = params['oversold']
        
        if len(data) < k_period:
            return 'HOLD'
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=k_period, 
                                       slowk_period=d_period, slowd_period=d_period)
        else:
            # Спрощена реалізація Stochastic
            lowest_low = pd.Series(low).rolling(window=k_period).min().values
            highest_high = pd.Series(high).rolling(window=k_period).max().values
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            slowk = pd.Series(k_percent).rolling(window=d_period).mean().values
            slowd = pd.Series(slowk).rolling(window=d_period).mean().values
        
        if len(slowk) < 2:
            return 'HOLD'
        
        current_k = slowk[-1]
        current_d = slowd[-1]
        previous_k = slowk[-2]
        previous_d = slowd[-2]
        
        # Oversold crossover
        if previous_k <= oversold and current_k > oversold and current_k > current_d:
            return 'BUY'
        # Overbought crossover
        elif previous_k >= overbought and current_k < overbought and current_k < current_d:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_williams_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Williams %R стратегія"""
        period = params['williams_period']
        overbought = params['overbought']
        oversold = params['oversold']
        
        if len(data) < period:
            return 'HOLD'
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if TALIB_AVAILABLE:
            williams = talib.WILLR(high, low, close, timeperiod=period)
        else:
            # Спрощена реалізація Williams %R
            highest_high = pd.Series(high).rolling(window=period).max().values
            lowest_low = pd.Series(low).rolling(window=period).min().values
            williams = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        if len(williams) < 2:
            return 'HOLD'
        
        current_williams = williams[-1]
        previous_williams = williams[-2]
        
        # Oversold reversal
        if previous_williams <= oversold and current_williams > oversold:
            return 'BUY'
        # Overbought reversal
        elif previous_williams >= overbought and current_williams < overbought:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_cci_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Commodity Channel Index стратегія"""
        period = params['cci_period']
        threshold = params['cci_threshold']
        
        if len(data) < period:
            return 'HOLD'
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if TALIB_AVAILABLE:
            cci = talib.CCI(high, low, close, timeperiod=period)
        else:
            # Спрощена реалізація CCI
            typical_price = (high + low + close) / 3
            sma_tp = pd.Series(typical_price).rolling(window=period).mean().values
            mean_deviation = pd.Series(typical_price).rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        if len(cci) < 2:
            return 'HOLD'
        
        current_cci = cci[-1]
        previous_cci = cci[-2]
        
        # CCI crossover
        if previous_cci <= -threshold and current_cci > -threshold:
            return 'BUY'
        elif previous_cci >= threshold and current_cci < threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_multi_timeframe_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Multi-timeframe стратегія"""
        # Спрощена реалізація для симуляції
        if len(data) < 50:
            return 'HOLD'
        
        # Симуляція сигналів з різних таймфреймів
        daily_signal = self._generate_momentum_signal(data, {'lookback_period': 20, 'momentum_threshold': 0.02})
        
        # Симуляція годинного сигналу
        hourly_data = data.tail(24) if len(data) >= 24 else data
        hourly_signal = self._generate_momentum_signal(hourly_data, {'lookback_period': 10, 'momentum_threshold': 0.015})
        
        # Консенсус сигналів
        if daily_signal == 'BUY' and hourly_signal == 'BUY':
            return 'BUY'
        elif daily_signal == 'SELL' and hourly_signal == 'SELL':
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_sector_rotation_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Sector rotation стратегія"""
        lookback = params['lookback_period']
        threshold = params['sector_momentum_threshold']
        
        if len(data) < lookback:
            return 'HOLD'
        
        recent_returns = data['close'].pct_change().tail(lookback)
        momentum = recent_returns.mean()
        
        if momentum > threshold:
            return 'BUY'
        elif momentum < -threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _generate_pairs_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Pairs trading стратегія"""
        # Спрощена реалізація
        if len(data) < 50:
            return 'HOLD'
        
        # Розрахунок z-score для ціни
        prices = data['close']
        mean_price = prices.tail(20).mean()
        std_price = prices.tail(20).std()
        current_price = prices.iloc[-1]
        
        zscore = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        entry_threshold = params['zscore_entry']
        exit_threshold = params['zscore_exit']
        
        if abs(zscore) > entry_threshold:
            return 'BUY' if zscore < 0 else 'SELL'
        elif abs(zscore) < exit_threshold:
            return 'SELL' if zscore > 0 else 'BUY'
        else:
            return 'HOLD'
    
    def _generate_volatility_mean_reversion_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> str:
        """Volatility mean reversion стратегія"""
        window = params['volatility_window']
        percentile = params['volatility_percentile']
        
        if len(data) < window:
            return 'HOLD'
        
        returns = data['close'].pct_change().tail(window)
        current_volatility = returns.std()
        
        # Історичні волатильності
        historical_vols = []
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            historical_vols.append(window_returns.std())
        
        if not historical_vols:
            return 'HOLD'
        
        volatility_percentile = np.percentile(historical_vols, percentile)
        
        # Низька волатильність -> очікуємо вибух
        if current_volatility < volatility_percentile:
            return 'BUY'
        # Висока волатильність -> очікуємо спад
        elif current_volatility > np.percentile(historical_vols, 100 - percentile):
            return 'SELL'
        else:
            return 'HOLD'
    
    def run(self) -> Dict[str, Any]:
        """Запуск розширеного бектестингу"""
        self.logger.info("Starting comprehensive backtesting with advanced strategies...")
        
        # Оновлення конфігурації для розширених тікерів
        original_tickers = self.config.data.tickers.copy()
        
        try:
            # Використовуємо розширені тікери якщо не вказано інакше
            if len(original_tickers) <= 5:  # Якщо мало тікерів, розширюємо
                all_extended_tickers = []
                for sector_tickers in self.extended_tickers.values():
                    all_extended_tickers.extend(sector_tickers)
                self.config.data.tickers = list(set(all_extended_tickers))[:30]  # Обмежуємо до 30
            
            # Запуск базового бектестингу з розширеними стратегіями
            results = super().run()
            
            # Додавання extended analytics
            results['comprehensive_analysis'] = self._generate_comprehensive_analysis(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive backtesting failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Comprehensive backtesting process failed'
            }
        finally:
            # Відновлення оригінальних тікерів
            self.config.data.tickers = original_tickers
    
    def _generate_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерація розширеного аналізу результатів"""
        analysis = {
            'strategy_performance_matrix': {},
            'risk_adjusted_metrics': {},
            'market_regime_analysis': {},
            'optimization_suggestions': []
        }
        
        if results.get('status') != 'success':
            return analysis
        
        backtest_results = results.get('backtest_results', {})
        
        # Матриця продуктивності стратегій
        for strategy_name, strategy_result in backtest_results.items():
            if strategy_result.get('status') == 'success':
                metrics = strategy_result.get('metrics', {})
                analysis['strategy_performance_matrix'][strategy_name] = {
                    'return': metrics.get('total_return', 0),
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('winning_trades', 0) / max(metrics.get('total_trades', 1), 1),
                    'profit_factor': metrics.get('profit_factor', 0)
                }
        
        # Ризик-скориговані метрики
        all_returns = [m['return'] for m in analysis['strategy_performance_matrix'].values()]
        all_sharpes = [m['sharpe'] for m in analysis['strategy_performance_matrix'].values()]
        
        if all_returns and all_sharpes:
            analysis['risk_adjusted_metrics'] = {
                'best_risk_adjusted_strategy': max(analysis['strategy_performance_matrix'].keys(), 
                                                 key=lambda k: analysis['strategy_performance_matrix'][k]['sharpe']),
                'average_sharpe_ratio': np.mean(all_sharpes),
                'sharpe_ratio_std': np.std(all_sharpes),
                'return_to_drawdown_ratio': np.mean([abs(m['return'] / max(m['max_drawdown'], 0.01)) 
                                                   for m in analysis['strategy_performance_matrix'].values()])
            }
        
        # Рекомендації по оптимізації
        if analysis['strategy_performance_matrix']:
            best_strategy = max(analysis['strategy_performance_matrix'].items(), key=lambda x: x[1]['return'])
            worst_strategy = min(analysis['strategy_performance_matrix'].items(), key=lambda x: x[1]['return'])
            
            analysis['optimization_suggestions'] = [
                f"Focus on {best_strategy[0]} - highest return ({best_strategy[1]['return']:.2%})",
                f"Consider improving {worst_strategy[0]} - lowest return ({worst_strategy[1]['return']:.2%})",
                f"Implement dynamic position sizing based on volatility",
                f"Add stop-loss mechanisms to reduce max drawdown"
            ]
        
        return analysis
