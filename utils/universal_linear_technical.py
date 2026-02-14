# utils/universal_linear_technical.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("UniversalLinearTechnical")


class UniversalLinearTechnical:
    """
    Універсальний лінійний калькулятор технічних індикаторів
    Обчислює індикатори для всіх тікерів одночасно без groupby
    """
    
    def __init__(self):
        self.logger = logging.getLogger("UniversalLinearTechnical")
        
        # Кеш для результатів
        self.cache = {}
        self.cache_max_size = 1000
        
        # Підтримувані індикатори
        self.indicator_functions = {
            'sma': self._calculate_sma_linear,
            'ema': self._calculate_ema_linear,
            'rsi': self._calculate_rsi_linear,
            'macd': self._calculate_macd_linear,
            'bollinger_upper': self._calculate_bollinger_upper_linear,
            'bollinger_lower': self._calculate_bollinger_lower_linear,
            'bollinger_width': self._calculate_bollinger_width_linear,
            'atr': self._calculate_atr_linear,
            'momentum': self._calculate_momentum_linear,
            'roc': self._calculate_roc_linear,
            'volatility': self._calculate_volatility_linear,
            'williams_r': self._calculate_williams_r_linear,
            'stochastic_k': self._calculate_stochastic_k_linear,
            'stochastic_d': self._calculate_stochastic_d_linear,
            'cci': self._calculate_cci_linear,
            'adx': self._calculate_adx_linear,
            'mfi': self._calculate_mfi_linear,
            'obv': self._calculate_obv_linear,
            'vwap': self._calculate_vwap_linear,
            'volume_sma': self._calculate_volume_sma_linear,
            'volume_ratio': self._calculate_volume_ratio_linear,
            'price_change': self._calculate_price_change_linear,
            'log_return': self._calculate_log_return_linear,
            'cumulative_return': self._calculate_cumulative_return_linear,
            'high_low_ratio': self._calculate_high_low_ratio_linear,
            'close_position': self._calculate_close_position_linear,
            'price_position': self._calculate_price_position_linear
        }
        
        self.logger.info("UniversalLinearTechnical initialized")
    
    def calculate_all_indicators_for_all_tickers(self, data: pd.DataFrame, 
                                                timeframe: str = None,
                                                indicators: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Обчислює всі індикатори для всіх тікерів лінійно
        
        Args:
            data: DataFrame з тікер-специфічними колонками
            timeframe: Таймфрейм (якщо None - знайде автоматично)
            indicators: Список індикаторів (якщо None - всі)
            
        Returns:
            Dict: {ticker: {indicator: value}}
        """
        if indicators is None:
            indicators = list(self.indicator_functions.keys())
        
        # Створюємо ключ кешу
        cache_key = f"all_{timeframe}_{hash(str(data.columns.tobytes()))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Знаходимо всі тікери та таймфрейми
        ticker_timeframes = self._find_ticker_timeframes(data, timeframe)
        
        # Ініціалізуємо результати
        all_results = {}
        
        for ticker, tf in ticker_timeframes.items():
            all_results[ticker] = {}
        
        # Обчислюємо кожен індикатор
        for indicator in indicators:
            if indicator not in self.indicator_functions:
                self.logger.warning(f"Unknown indicator: {indicator}")
                continue
            
            try:
                indicator_results = self.indicator_functions[indicator](data, ticker_timeframes)
                
                # Додаємо результати до загального словника
                for ticker, value in indicator_results.items():
                    if ticker in all_results:
                        all_results[ticker][indicator] = value
                
            except Exception as e:
                self.logger.error(f"Error calculating {indicator}: {e}")
                continue
        
        # Кешуємо результат
        if len(self.cache) >= self.cache_max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = all_results
        
        return all_results
    
    def calculate_indicator_for_all_tickers(self, data: pd.DataFrame, 
                                             indicator: str, 
                                             window: int = None,
                                             timeframe: str = None,
                                             **kwargs) -> Dict[str, float]:
        """
        Обчислює один індикатор для всіх тікерів
        
        Args:
            data: DataFrame з даними
            indicator: Тип індикатора
            window: Вікно (якщо потрібно)
            timeframe: Таймфрейм
            **kwargs: Додаткові параметри
            
        Returns:
            Dict: {ticker: indicator_value}
        """
        if indicator not in self.indicator_functions:
            self.logger.error(f"Unknown indicator: {indicator}")
            return {}
        
        # Знаходимо тікери та таймфрейми
        ticker_timeframes = self._find_ticker_timeframes(data, timeframe)
        
        # Обчислюємо індикатор
        try:
            results = self.indicator_functions[indicator](data, ticker_timeframes, window, **kwargs)
            return results
        except Exception as e:
            self.logger.error(f"Error calculating {indicator}: {e}")
            return {}
    
    def _find_ticker_timeframes(self, data: pd.DataFrame, timeframe: str = None) -> Dict[str, str]:
        """
        Знаходить всі тікери та їхні таймфрейми в даних
        
        Args:
            data: DataFrame з даними
            timeframe: Бажаний таймфрейм
            
        Returns:
            Dict: {ticker: timeframe}
        """
        ticker_timeframes = {}
        
        # Знаходимо всі close колонки
        close_columns = [col for col in data.columns if col.endswith('_close')]
        
        for close_col in close_columns:
            # Витягуємо тікер і таймфрейм
            parts = close_col.split('_')
            if len(parts) >= 2:
                ticker = parts[0]
                tf = '_'.join(parts[1:-1])  # все між тікером і 'close'
                
                # Фільтруємо за бажаним таймфреймом
                if timeframe is None or tf == timeframe:
                    ticker_timeframes[ticker] = tf
        
        return ticker_timeframes
    
    # Лінійні функції індикаторів
    def _calculate_sma_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 20, **kwargs) -> Dict[str, float]:
        """Лінійний SMA для всіх тікерів"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            sma = prices.rolling(window=window, min_periods=1).mean()
            if not sma.empty:
                results[ticker] = sma.iloc[-1]
        
        return results
    
    def _calculate_ema_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                             window: int = 12, **kwargs) -> Dict[str, float]:
        """Лінійний EMA для всіх тікерів"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            ema = prices.ewm(span=window, adjust=False, min_periods=1).mean()
            if not ema.empty:
                results[ticker] = ema.iloc[-1]
        
        return results
    
    def _calculate_rsi_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний RSI для всіх тікерів"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if not rsi.empty:
                results[ticker] = rsi.iloc[-1]
        
        return results
    
    def _calculate_macd_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                              slow: int = 26, fast: int = 12, signal: int = 9, **kwargs) -> Dict[str, float]:
        """Лінійний MACD для всіх тікерів"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < slow:
                continue
            
            ema_fast = prices.ewm(span=fast, adjust=False, min_periods=1).mean()
            ema_slow = prices.ewm(span=slow, adjust=False, min_periods=1).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
            
            if not macd_line.empty and not signal_line.empty:
                results[ticker] = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        return results
    
    def _calculate_bollinger_upper_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                         window: int = 20, std_dev: float = 2, **kwargs) -> Dict[str, float]:
        """Лінійні верхні Bollinger Bands"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            sma = prices.rolling(window=window, min_periods=1).mean()
            std = prices.rolling(window=window, min_periods=1).std()
            upper = sma + (std * std_dev)
            
            if not upper.empty:
                results[ticker] = upper.iloc[-1]
        
        return results
    
    def _calculate_bollinger_lower_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                         window: int = 20, std_dev: float = 2, **kwargs) -> Dict[str, float]:
        """Лінійні нижні Bollinger Bands"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            sma = prices.rolling(window=window, min_periods=1).mean()
            std = prices.rolling(window=window, min_periods=1).std()
            lower = sma - (std * std_dev)
            
            if not lower.empty:
                results[ticker] = lower.iloc[-1]
        
        return results
    
    def _calculate_bollinger_width_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                        window: int = 20, std_dev: float = 2, **kwargs) -> Dict[str, float]:
        """Лінійна ширина Bollinger Bands"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            sma = prices.rolling(window=window, min_periods=1).mean()
            std = prices.rolling(window=window, min_periods=1).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            width = (upper - lower) / sma
            
            if not width.empty:
                results[ticker] = width.iloc[-1]
        
        return results
    
    def _calculate_atr_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний ATR"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window, min_periods=1).mean()
            
            if not atr.empty:
                results[ticker] = atr.iloc[-1]
        
        return results
    
    def _calculate_momentum_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                 window: int = 5, **kwargs) -> Dict[str, float]:
        """Лінійний моментум"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            momentum = prices.pct_change(periods=window)
            if not momentum.empty:
                results[ticker] = momentum.iloc[-1]
        
        return results
    
    def _calculate_roc_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 10, **kwargs) -> Dict[str, float]:
        """Лінійний Rate of Change"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            roc = ((prices / prices.shift(window)) - 1) * 100
            if not roc.empty:
                results[ticker] = roc.iloc[-1]
        
        return results
    
    def _calculate_volatility_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                    window: int = 20, **kwargs) -> Dict[str, float]:
        """Лінійна волатильність"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            volatility = prices.rolling(window=window, min_periods=1).std()
            if not volatility.empty:
                results[ticker] = volatility.iloc[-1]
        
        return results
    
    def _calculate_williams_r_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                   window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний Williams %R"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            highest_high = high.rolling(window=window, min_periods=1).max()
            lowest_low = low.rolling(window=window, min_periods=1).min()
            
            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            if not wr.empty:
                results[ticker] = wr.iloc[-1]
        
        return results
    
    def _calculate_stochastic_k_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                     window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний Stochastic %K"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            lowest_low = low.rolling(window=window, min_periods=1).min()
            highest_high = high.rolling(window=window, min_periods=1).max()
            
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            if not k.empty:
                results[ticker] = k.iloc[-1]
        
        return results
    
    def _calculate_stochastic_d_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                     window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний Stochastic %D"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            lowest_low = low.rolling(window=window, min_periods=1).min()
            highest_high = high.rolling(window=window, min_periods=1).max()
            
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d = k.rolling(window=3, min_periods=1).mean()
            
            if not d.empty:
                results[ticker] = d.iloc[-1]
        
        return results
    
    def _calculate_cci_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 20, **kwargs) -> Dict[str, float]:
        """Лінійний Commodity Channel Index"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=window, min_periods=1).mean()
            mean_deviation = (typical_price - sma_tp).abs().rolling(window=window, min_periods=1).mean()
            
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            if not cci.empty:
                results[ticker] = cci.iloc[-1]
        
        return results
    
    def _calculate_adx_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний ADX"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            plus_dm = high - high.shift()
            minus_dm = low.shift() - low
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            plus_dm_smooth = plus_dm.rolling(window=window, min_periods=1).mean()
            minus_dm_smooth = minus_dm.rolling(window=window, min_periods=1).mean()
            tr_smooth = true_range.rolling(window=window, min_periods=1).mean()
            
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(window=window, min_periods=1).mean()
            
            if not adx.empty:
                results[ticker] = adx.iloc[-1]
        
        return results
    
    def _calculate_mfi_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            window: int = 14, **kwargs) -> Dict[str, float]:
        """Лінійний Money Flow Index"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            volume_col = f'{ticker}_{tf}_volume'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col, volume_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            volume = data[volume_col].dropna()
            
            if len(close) < window:
                continue
            
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_mf = money_flow.where(money_flow > 0, 0)
            negative_mf = -money_flow.where(money_flow < 0, 0)
            
            positive_mf_sum = positive_mf.rolling(window=window, min_periods=1).sum()
            negative_mf_sum = negative_mf.rolling(window=window, min_periods=1).sum()
            
            mfi = 100 * (positive_mf_sum / (positive_mf_sum + negative_mf_sum))
            if not mfi.empty:
                results[ticker] = mfi.iloc[-1]
        
        return results
    
    def _calculate_obv_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                            **kwargs) -> Dict[str, float]:
        """Лінійний On-Balance Volume"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            volume_col = f'{ticker}_{tf}_volume'
            
            if not all(col in data.columns for col in [close_col, volume_col]):
                continue
            
            close = data[close_col].dropna()
            volume = data[volume_col].dropna()
            
            price_change = close.diff()
            obv = volume.where(price_change > 0, -volume).where(price_change < 0, volume)
            obv = obv.cumsum()
            
            if not obv.empty:
                results[ticker] = obv.iloc[-1]
        
        return results
    
    def _calculate_vwap_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                             **kwargs) -> Dict[str, float]:
        """Лінійний Volume Weighted Average Price"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            volume_col = f'{ticker}_{tf}_volume'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col, volume_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            volume = data[volume_col].dropna()
            
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            
            if not vwap.empty:
                results[ticker] = vwap.iloc[-1]
        
        return results
    
    def _calculate_volume_sma_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                   window: int = 20, **kwargs) -> Dict[str, float]:
        """Лінійний Volume SMA"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            volume_col = f'{ticker}_{tf}_volume'
            if volume_col not in data.columns:
                continue
            
            volume = data[volume_col].dropna()
            if len(volume) < window:
                continue
            
            volume_sma = volume.rolling(window=window, min_periods=1).mean()
            if not volume_sma.empty:
                results[ticker] = volume_sma.iloc[-1]
        
        return results
    
    def _calculate_volume_ratio_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                     window: int = 20, **kwargs) -> Dict[str, float]:
        """Лінійний Volume Ratio"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            volume_col = f'{ticker}_{tf}_volume'
            if volume_col not in data.columns:
                continue
            
            volume = data[volume_col].dropna()
            if len(volume) < window:
                continue
            
            current_volume = volume.iloc[-1]
            avg_volume = volume.rolling(window=window, min_periods=1).mean().iloc[-1]
            
            if avg_volume > 0:
                results[ticker] = current_volume / avg_volume
        
        return results
    
    def _calculate_price_change_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                     **kwargs) -> Dict[str, float]:
        """Лінійна зміна ціни"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < 2:
                continue
            
            price_change = prices.pct_change()
            if not price_change.empty:
                results[ticker] = price_change.iloc[-1]
        
        return results
    
    def _calculate_log_return_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                   **kwargs) -> Dict[str, float]:
        """Лінійний логарифмічний дохід"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < 2:
                continue
            
            log_return = np.log(prices / prices.shift(1))
            if not log_return.empty:
                results[ticker] = log_return.iloc[-1]
        
        return results
    
    def _calculate_cumulative_return_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                          **kwargs) -> Dict[str, float]:
        """Лінійний кумулятивний дохід"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < 2:
                continue
            
            cumulative_return = (prices / prices.iloc[0] - 1) * 100
            if not cumulative_return.empty:
                results[ticker] = cumulative_return.iloc[-1]
        
        return results
    
    def _calculate_high_low_ratio_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                       **kwargs) -> Dict[str, float]:
        """Лінійне співвідношення high/low"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [high_col, low_col]):
                continue
            
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(high) < 1:
                continue
            
            ratio = high / low
            if not ratio.empty:
                results[ticker] = ratio.iloc[-1]
        
        return results
    
    def _calculate_close_position_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                       window: int = 20, **kwargs) -> Dict[str, float]:
        """Лінійна позиція ціни в діапазоні"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            high_col = f'{ticker}_{tf}_high'
            low_col = f'{ticker}_{tf}_low'
            
            if not all(col in data.columns for col in [close_col, high_col, low_col]):
                continue
            
            close = data[close_col].dropna()
            high = data[high_col].dropna()
            low = data[low_col].dropna()
            
            if len(close) < window:
                continue
            
            highest_high = high.rolling(window=window, min_periods=1).max()
            lowest_low = low.rolling(window=window, min_periods=1).min()
            
            position = (close - lowest_low) / (highest_high - lowest_low)
            if not position.empty:
                results[ticker] = position.iloc[-1]
        
        return results
    
    def _calculate_price_position_linear(self, data: pd.DataFrame, ticker_timeframes: Dict[str, str], 
                                       window: int = 100, **kwargs) -> Dict[str, float]:
        """Лінійна позиція ціни відносно історичних максимумів/мінімумів"""
        results = {}
        
        for ticker, tf in ticker_timeframes.items():
            close_col = f'{ticker}_{tf}_close'
            if close_col not in data.columns:
                continue
            
            prices = data[close_col].dropna()
            if len(prices) < window:
                continue
            
            highest = prices.rolling(window=window, min_periods=1).max()
            lowest = prices.rolling(window=window, min_periods=1).min()
            
            position = (prices - lowest) / (highest - lowest)
            if not position.empty:
                results[ticker] = position.iloc[-1]
        
        return results


# Приклад використання
if __name__ == "__main__":
    # Створюємо калькулятор
    calculator = UniversalLinearTechnical()
    
    # Припустимо у нас є DataFrame з даними
    # data має колонки: TSLA_15m_close, NVDA_15m_close, AAPL_15m_close, etc.
    
    # Обчислюємо всі індикатори для всіх тікерів
    all_indicators = calculator.calculate_all_indicators_for_all_tickers(data, timeframe='15m')
    print(f"Indicators for TSLA: {all_indicators.get('TSLA', {})}")
    
    # Обчислюємо один індикатор
    rsi_values = calculator.calculate_indicator_for_all_tickers(data, 'rsi', 14, '15m')
    print(f"RSI values: {rsi_values}")
    
    # Обчислюємо SMA з кастомним вікном
    sma_values = calculator.calculate_indicator_for_all_tickers(data, 'sma', 50, '15m')
    print(f"SMA 50 values: {sma_values}")
