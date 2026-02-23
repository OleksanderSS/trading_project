# utils/universal_indicators.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("UniversalIndicators")


class UniversalIndicatorCalculator:
    """
    Універсальний калькулятор індикаторів для всіх тікерів одночасно
    """
    
    def __init__(self):
        self.logger = logging.getLogger("UniversalIndicators")
    
    def calculate_universal_indicators(self, data: pd.DataFrame, indicator_type: str, 
                                       window: int = 20) -> Dict[str, pd.Series]:
        """
        Обчислює універсальні індикатори для всіх тікерів одночасно
        
        Args:
            data: DataFrame з даними (тікер-специфічні колонки)
            indicator_type: Тип індикатора ('sma', 'ema', 'rsi', 'macd', 'momentum', 'volatility')
            window: Вікно для розрахунку
            
        Returns:
            Dict: {ticker_name: pd.Series з індикатором}
        """
        indicators = {}
        
        # Знаходимо всі тікер-специфічні колонки для ціни
        price_columns = [col for col in data.columns if col.endswith('_close')]
        
        for price_col in price_columns:
            # Витягуємо назву тікера
            ticker = price_col.split('_')[0]
            
            # Отримуємо ціни для цього тікера
            prices = data[price_col].dropna()
            
            if len(prices) < window:
                continue
            
            # Обчислюємо індикатор залежно від типу
            if indicator_type == 'sma':
                indicator = prices.rolling(window=window, min_periods=1).mean()
            elif indicator_type == 'ema':
                indicator = prices.ewm(span=window, adjust=False, min_periods=1).mean()
            elif indicator_type == 'rsi':
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / loss
                indicator = 100 - (100 / (1 + rs))
            elif indicator_type == 'momentum':
                indicator = prices.pct_change(periods=window)
            elif indicator_type == 'volatility':
                indicator = prices.rolling(window=window, min_periods=1).std()
            elif indicator_type == 'bollinger_upper':
                sma = prices.rolling(window=window, min_periods=1).mean()
                std = prices.rolling(window=window, min_periods=1).std()
                indicator = sma + (std * 2)
            elif indicator_type == 'bollinger_lower':
                sma = prices.rolling(window=window, min_periods=1).mean()
                std = prices.rolling(window=window, min_periods=1).std()
                indicator = sma - (std * 2)
            elif indicator_type == 'atr':
                if f'{ticker}_high' in data.columns and f'{ticker}_low' in data.columns:
                    high = data[f'{ticker}_high']
                    low = data[f'{ticker}_low']
                    close = prices
                    tr1 = high - low
                    tr2 = (high - close.shift()).abs()
                    tr3 = (low - close.shift()).abs()
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    indicator = true_range.rolling(window=window, min_periods=1).mean()
                else:
                    continue
            elif indicator_type == 'volume_sma':
                if f'{ticker}_volume' in data.columns:
                    volume = data[f'{ticker}_volume']
                    indicator = volume.rolling(window=window, min_periods=1).mean()
                else:
                    continue
            elif indicator_type == 'price_change':
                indicator = prices.pct_change()
            elif indicator_type == 'log_return':
                indicator = np.log(prices / prices.shift(1))
            elif indicator_type == 'cumulative_return':
                indicator = (prices / prices.iloc[0] - 1) * 100
            else:
                continue
            
            indicators[ticker] = indicator
        
        return indicators
    
    def get_universal_context(self, data: pd.DataFrame, 
                                 indicators_config: List[Dict] = None) -> Dict[str, Any]:
        """
        Отримує універсальний контекст для всіх тікерів
        
        Args:
            data: DataFrame з даними
            indicators_config: Конфігурація індикаторів
            
        Returns:
            Dict: Універсальний контекст
        """
        if indicators_config is None:
            indicators_config = [
                {'type': 'sma', 'window': 20, 'weight': 0.1},
                {'type': 'ema', 'window': 12, 'weight': 0.1},
                {'type': 'rsi', 'window': 14, 'weight': 0.15},
                {'type': 'momentum', 'window': 5, 'weight': 0.1},
                {'type': 'volatility', 'window': 20, 'weight': 0.15},
                {'type': 'bollinger_upper', 'window': 20, 'weight': 0.05},
                {'type': 'bollinger_lower', 'window': 20, 'weight': 0.05},
                {'type': 'atr', 'window': 14, 'weight': 0.1},
                {'type': 'volume_sma', 'window': 20, 'weight': 0.05},
                {'type': 'price_change', 'window': 1, 'weight': 0.1},
                {'type': 'log_return', 'window': 1, 'weight': 0.05},
                {'type': 'cumulative_return', 'window': 0, 'weight': 0.15}
            ]
        
        universal_context = {
            'timestamp': datetime.now().isoformat(),
            'tickers': {},
            'market_overview': {},
            'top_performers': {},
            'bottom_performers': {},
            'market_signals': {}
        }
        
        # Обчислюємо всі індикатори для всіх тікерів
        all_indicators = {}
        for config in indicators_config:
            indicator_type = config['type']
            window = config['window']
            indicators = self.calculate_universal_indicators(data, indicator_type, window)
            
            for ticker, indicator_series in indicators.items():
                if ticker not in all_indicators:
                    all_indicators[ticker] = {}
                
                all_indicators[ticker][indicator_type] = {
                    'series': indicator_series,
                    'current_value': indicator_series.iloc[-1] if not indicator_series.empty else None,
                    'previous_value': indicator_series.iloc[-2] if len(indicator_series) > 1 else None,
                    'weight': config['weight']
                }
        
        # Обробляємо кожен тікер
        ticker_scores = {}
        for ticker, indicators_data in all_indicators.items():
            ticker_context = {
                'indicators': {},
                'overall_score': 0,
                'signal': 'neutral',
                'performance': {}
            }
            
            total_score = 0
            total_weight = 0
            
            # Розраховуємо скор для кожного індикатора
            for indicator_type, indicator_data in indicators_data.items():
                current_value = indicator_data['current_value']
                previous_value = indicator_data['previous_value']
                weight = indicator_data['weight']
                
                if current_value is not None and previous_value is not None:
                    # Просте порівняння
                    if current_value > previous_value:
                        comparison = 1
                    elif current_value < previous_value:
                        comparison = -1
                    else:
                        comparison = 0
                    
                    # Специфічна логіка для різних індикаторів
                    if indicator_type in ['rsi', 'volatility', 'atr']:
                        # Для RSI, волатильності - екстремальні значення важливіші
                        if indicator_type == 'rsi':
                            if current_value > 70:  # Overbought
                                score = -1 * weight
                            elif current_value < 30:  # Oversold
                                score = 1 * weight
                            else:
                                score = comparison * weight * 0.5
                        elif indicator_type == 'volatility':
                            # Висока волатильність може бути добре або погано
                            score = comparison * weight * 0.7
                        else:  # ATR
                            score = comparison * weight * 0.6
                    elif indicator_type in ['momentum', 'price_change', 'log_return']:
                        # Для моментуму - позитивні зміни кращі
                        score = comparison * weight * 1.2
                    elif indicator_type in ['sma', 'ema']:
                        # Для ковзних середніх - порівнюємо з ціною
                        price_col = f'{ticker}_close'
                        if price_col in data.columns:
                            current_price = data[price_col].iloc[-1]
                            if current_value is not None:
                                price_vs_ma = (current_price - current_value) / current_value
                                score = np.sign(price_vs_ma) * weight
                            else:
                                score = 0
                        else:
                            score = 0
                    else:
                        # Для інших індикаторів
                        score = comparison * weight
                    
                    ticker_context['indicators'][indicator_type] = {
                        'current_value': current_value,
                        'previous_value': previous_value,
                        'comparison': comparison,
                        'score': score,
                        'weight': weight
                    }
                    
                    total_score += score
                    total_weight += weight
            
            # Розраховуємо загальний скор і сигнал
            if total_weight > 0:
                ticker_context['overall_score'] = total_score / total_weight
                
                if ticker_context['overall_score'] > 0.1:
                    ticker_context['signal'] = 'bullish'
                elif ticker_context['overall_score'] < -0.1:
                    ticker_context['signal'] = 'bearish'
                else:
                    ticker_context['signal'] = 'neutral'
            
            # Додаємо перформанс дані
            if f'{ticker}_close' in data.columns:
                prices = data[f'{ticker}_close'].dropna()
                if len(prices) > 1:
                    ticker_context['performance'] = {
                        'current_price': prices.iloc[-1],
                        'previous_price': prices.iloc[-2],
                        'price_change_pct': (prices.iloc[-1] / prices.iloc[-2] - 1) * 100,
                        'cumulative_return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) > 0 else 0
                    }
            
            universal_context['tickers'][ticker] = ticker_context
            ticker_scores[ticker] = ticker_context['overall_score']
        
        # Ранжуємо тікери
        sorted_tickers = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Топ перформери
        universal_context['top_performers'] = {
            'bullish': [ticker for ticker, score in sorted_tickers[:5] if score > 0.1],
            'bearish': [ticker for ticker, score in sorted_tickers[-5:] if score < -0.1],
            'neutral': [ticker for ticker, score in sorted_tickers if -0.1 <= score <= 0.1][:5]
        }
        
        # Загальний огляд ринку
        bullish_count = sum(1 for ticker, score in ticker_scores.items() if score > 0.1)
        bearish_count = sum(1 for ticker, score in ticker_scores.items() if score < -0.1)
        total_count = len(ticker_scores)
        
        universal_context['market_overview'] = {
            'total_tickers': total_count,
            'bullish_tickers': bullish_count,
            'bearish_tickers': bearish_count,
            'neutral_tickers': total_count - bullish_count - bearish_count,
            'market_sentiment': 'bullish' if bullish_count > bearish_count else 'bearish' if bearish_count > bullish_count else 'neutral',
            'average_score': sum(ticker_scores.values()) / total_count if total_count > 0 else 0
        }
        
        # Ринкові сигнали
        universal_context['market_signals'] = {
            'strong_bullish': [ticker for ticker, score in sorted_tickers[:3] if score > 0.2],
            'strong_bearish': [ticker for ticker, score in sorted_tickers[-3:] if score < -0.2],
            'reversal_candidates': [ticker for ticker, score in sorted_tickers if abs(score) > 0.15][:5]
        }
        
        return universal_context
    
    def get_ticker_universe_analysis(self, data: pd.DataFrame, 
                                        universe_tickers: List[str] = None) -> Dict[str, Any]:
        """
        Аналізує всесвіт тікерів
        
        Args:
            data: DataFrame з даними
            universe_tickers: Список тікерів для аналізу
            
        Returns:
            Dict: Аналіз всесвіту
        """
        if universe_tickers is None:
            # Автоматично знаходимо всі тікери в даних
            universe_tickers = list(set([col.split('_')[0] for col in data.columns if '_' in col]))
        
        # Фільтруємо дані тільки для тікерів зі всесвіту
        universe_data = data.copy()
        universe_columns = []
        
        for col in universe_data.columns:
            if '_' in col:
                ticker = col.split('_')[0]
                if ticker in universe_tickers:
                    universe_columns.append(col)
        
        universe_data = universe_data[universe_columns]
        
        # Отримуємо універсальний контекст
        universe_context = self.get_universal_context(universe_data)
        
        # Додаємо специфічний аналіз всесвіту
        universe_analysis = {
            **universe_context,
            'universe_size': len(universe_tickers),
            'universe_performance': {},
            'sector_analysis': {},
            'correlation_analysis': {},
            'risk_metrics': {}
        }
        
        # Аналіз перформансу всесвіту
        all_returns = []
        for ticker, context in universe_context['tickers'].items():
            if 'performance' in context:
                perf = context['performance']
                all_returns.append(perf.get('price_change_pct', 0))
        
        if all_returns:
            universe_analysis['universe_performance'] = {
                'average_return': np.mean(all_returns),
                'return_std': np.std(all_returns),
                'positive_returns': sum(1 for r in all_returns if r > 0),
                'negative_returns': sum(1 for r in all_returns if r < 0),
                'best_performer': max(all_returns),
                'worst_performer': min(all_returns)
            }
        
        return universe_analysis
    
    def get_simple_indicator_analysis(self, data: pd.DataFrame, indicator_type: str, 
                                    window: int = 20) -> Dict[str, Any]:
        """
        Простий аналіз одного індикатора для всіх тікерів
        
        Args:
            data: DataFrame з даними
            indicator_type: Тип індикатора
            window: Вікно для розрахунку
            
        Returns:
            Dict: Аналіз індикатора
        """
        indicators = self.calculate_universal_indicators(data, indicator_type, window)
        
        analysis = {
            'indicator_type': indicator_type,
            'window': window,
            'tickers': {},
            'summary': {}
        }
        
        ticker_scores = []
        
        for ticker, indicator_series in indicators.items():
            if not indicator_series.empty:
                current_value = indicator_series.iloc[-1]
                previous_value = indicator_series.iloc[-2] if len(indicator_series) > 1 else None
                
                if previous_value is not None:
                    change_pct = ((current_value - previous_value) / previous_value) * 100
                    signal = 'bullish' if change_pct > 0 else 'bearish' if change_pct < 0 else 'neutral'
                else:
                    change_pct = 0
                    signal = 'neutral'
                
                analysis['tickers'][ticker] = {
                    'current_value': current_value,
                    'previous_value': previous_value,
                    'change_pct': change_pct,
                    'signal': signal,
                    'series_length': len(indicator_series)
                }
                
                if indicator_type in ['rsi']:
                    # RSI специфічна логіка
                    if current_value > 70:
                        analysis['tickers'][ticker]['condition'] = 'overbought'
                    elif current_value < 30:
                        analysis['tickers'][ticker]['condition'] = 'oversold'
                    else:
                        analysis['tickers'][ticker]['condition'] = 'normal'
                
                ticker_scores.append((ticker, change_pct))
        
        # Загальний підсумок
        if ticker_scores:
            analysis['summary'] = {
                'total_tickers': len(ticker_scores),
                'average_change': np.mean([score for _, score in ticker_scores]),
                'best_performer': max(ticker_scores, key=lambda x: x[1])[0],
                'worst_performer': min(ticker_scores, key=lambda x: x[1])[0],
                'bullish_count': sum(1 for _, score in ticker_scores if score > 0),
                'bearish_count': sum(1 for _, score in ticker_scores if score < 0),
                'neutral_count': sum(1 for _, score in ticker_scores if score == 0)
            }
        
        return analysis
