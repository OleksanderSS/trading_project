#!/usr/bin/env python3
"""
Спільні утиліти та базові компоненти для уникнення дублювання коду
Централізовані функції для всіх modules системи
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from functools import wraps, lru_cache
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

from config.trading_config import TradingConfig
from utils.enhanced_data_validator import DataValidator


class TechnicalIndicators:
    """Централізовані технічні індикатори"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Розрахунок MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Розрахунок Bollinger Bands"""
        rolling_mean = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper_band = rolling_mean + (rolling_std * std)
        lower_band = rolling_mean - (rolling_std * std)
        return upper_band, rolling_mean, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок ATR"""
        high_low = high - low
        high_close = np.abs(high.shift(1) - close)
        low_close = np.abs(low.shift(1) - close)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Розрахунок Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Розрахунок CCI"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci


class DataProcessor:
    """Централізована обробка data"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищення OHLCV data"""
        # Видалення дублікатів
        df = df.drop_duplicates()
        
        # Сортування по даті
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Перевірка OHLC логіки
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Видалення аномальних значень
        for col in ['open', 'high', 'low', 'close']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def fill_missing_data(self, df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """Заповнення пропущених data"""
        if method == 'forward':
            return df.fillna(method='ffill')
        elif method == 'backward':
            return df.fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate()
        else:
            return df.dropna()
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Нормалізація data"""
        df_normalized = df.copy()
        for col in columns:
            if col in df.columns:
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df_normalized
    
    def calculate_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """Розрахунок доходностей"""
        return prices.pct_change(periods)
    
    def calculate_log_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """Розрахунок логарифмічних доходностей"""
        return np.log(prices / prices.shift(periods))


class CacheManager:
    """Централізоване кешування"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_key(self, key: str, params: Dict[str, Any] = None) -> str:
        """Генерація ключа кешу"""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            key = f"{key}_{hashlib.md5(param_str.encode()).hexdigest()}"
        return key
    
    def get(self, key: str, params: Dict[str, Any] = None) -> Any:
        """Отримання data з кешу"""
        cache_key = self._get_cache_key(key, params)
        
        # Перевірка в пам'яті
        if cache_key in self.memory_cache:
            cache_data = self.memory_cache[cache_key]
            if self._is_cache_valid(cache_data):
                return cache_data['value']
            else:
                del self.memory_cache[cache_key]
        
        # Перевірка в файлі
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if self._is_cache_valid(cache_data):
                        self.memory_cache[cache_key] = cache_data
                        return cache_data['value']
                    else:
                        cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Перевірка чи не закінчився термін дії кешу"""
        if 'timestamp' not in cache_data or 'ttl' not in cache_data:
            return True  # Старий формат, вважаємо дійсним
        
        return time.time() - cache_data['timestamp'] < cache_data['ttl']
    
    def set(self, key: str, value: Any, params: Dict[str, Any] = None, ttl: int = 3600) -> None:
        """Збереження data в кеш"""
        cache_key = self._get_cache_key(key, params)
        
        # Збереження з timestamp для TTL
        cache_data = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        # Збереження в пам'ять
        self.memory_cache[cache_key] = cache_data
        
        # Збереження в файл
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def clear(self) -> None:
        """Очищення кешу"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")


class PerformanceMonitor:
    """Моніторинг продуктивності"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def time_execution(self, func_name: str):
        """Декоратор для вимірювання часу виконання"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                if func_name not in self.metrics:
                    self.metrics[func_name] = []
                self.metrics[func_name].append(execution_time)
                
                self.logger.debug(f"{func_name} executed in {execution_time:.4f} seconds")
                return result
            return wrapper
        return decorator
    
    def get_average_time(self, func_name: str) -> float:
        """Отримання середнього часу виконання"""
        if func_name in self.metrics and self.metrics[func_name]:
            return np.mean(self.metrics[func_name])
        return 0.0
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Отримання звіту про продуктивність"""
        report = {}
        for func_name, times in self.metrics.items():
            if times:
                report[func_name] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times),
                    'call_count': len(times)
                }
        return report


class ErrorHandler:
    """Централізована обробка помилок"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Обробка помилки"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Лічильник помилок
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Логування
        self.logger.error(f"Error in {context}: {error_type} - {error_message}")
        
        return {
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'timestamp': datetime.now(),
            'count': self.error_counts[error_type]
        }
    
    def retry_on_error(self, max_retries: int = 3, delay: float = 1.0):
        """Декоратор для повторних спроб"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries:
                            self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                            time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        else:
                            self.logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                
                raise last_error
            return wrapper
        return decorator
    
    def get_error_summary(self) -> Dict[str, int]:
        """Отримання підсумку помилок"""
        return self.error_counts.copy()


class FileManager:
    """Централізована робота з файлами"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.logger = logging.getLogger(__name__)
    
    def ensure_directory(self, dir_path: Union[str, Path]) -> Path:
        """Створення директорії якщо не існує"""
        path = Path(dir_path)
        if not isinstance(path, Path):
            path = self.base_dir / path
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Збереження JSON файлу"""
        file_path = self.ensure_directory(file_path).parent / Path(file_path).name
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            self.logger.info(f"Saved JSON to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")
            raise
    
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Завантаження JSON файлу"""
        file_path = self.base_dir / file_path if not isinstance(file_path, Path) else file_path
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded JSON from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, file_path: Union[str, Path], format: str = 'parquet') -> None:
        """Збереження DataFrame"""
        file_path = self.ensure_directory(file_path).parent / Path(file_path).name
        
        try:
            if format == 'parquet':
                df.to_parquet(file_path)
            elif format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved DataFrame to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to {file_path}: {e}")
            raise
    
    def load_dataframe(self, file_path: Union[str, Path], format: str = 'parquet') -> pd.DataFrame:
        """Завантаження DataFrame"""
        file_path = self.base_dir / file_path if not isinstance(file_path, Path) else file_path
        
        try:
            if format == 'parquet':
                df = pd.read_parquet(file_path)
            elif format == 'csv':
                df = pd.read_csv(file_path)
            elif format == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Loaded DataFrame from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load DataFrame from {file_path}: {e}")
            raise


class ParallelProcessor:
    """Централізована паралельна обробка"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_parallel(self, func, items: List[Any], **kwargs) -> List[Any]:
        """Паралельна обробка списку"""
        if not items:
            return []
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(func, item, **kwargs): item for item in items}
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {item}: {e}")
                    results.append(None)
        
        return results
    
    def process_dict_parallel(self, func, items: Dict[Any, Any], **kwargs) -> Dict[Any, Any]:
        """Паралельна обробка словника"""
        if not items:
            return {}
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {executor.submit(func, key, value, **kwargs): key for key, value in items.items()}
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    results[key] = result
                except Exception as e:
                    self.logger.error(f"Error processing {key}: {e}")
                    results[key] = None
        
        return results


# Глобальні екземпляри для використання в усьому проекті
technical_indicators = TechnicalIndicators()
performance_monitor = PerformanceMonitor()
error_handler = ErrorHandler()
file_manager = FileManager()


def get_technical_indicators() -> TechnicalIndicators:
    """Отримання екземпляру технічних індикаторів"""
    return technical_indicators


def get_performance_monitor() -> PerformanceMonitor:
    """Отримання екземпляру монітора продуктивності"""
    return performance_monitor


def get_error_handler(logger: logging.Logger = None) -> ErrorHandler:
    """Отримання екземпляру обробника помилок"""
    return ErrorHandler(logger)


def get_file_manager(base_dir: Path = None) -> FileManager:
    """Отримання екземпляру менеджера файлів"""
    return FileManager(base_dir)


# Декоратори для використання в усьому проекті
def monitor_performance(func_name: str = None):
    """Декоратор для моніторингу продуктивності"""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        return performance_monitor.time_execution(name)(func)
    return decorator


def handle_errors(context: str = ""):
    """Декоратор для обробки помилок"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = error_handler.handle_error(e, context or func.__name__)
                raise
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Декоратор для повторних спроб"""
    return error_handler.retry_on_error(max_retries, delay)


def cache_result(ttl: int = 3600, cache_manager: CacheManager = None):
    """Декоратор для кешування результатів"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_mgr = cache_manager or CacheManager()
            
            # Генерація ключа кешу
            key = f"{func.__module__}.{func.__name__}"
            params = {'args': str(args), 'kwargs': str(kwargs)}
            
            # Спроба отримати з кешу
            cached_result = cache_mgr.get(key, params)
            if cached_result is not None:
                return cached_result
            
            # Виконання функції та кешування результату
            result = func(*args, **kwargs)
            cache_mgr.set(key, result, params, ttl)
            
            return result
        return wrapper
    return decorator
