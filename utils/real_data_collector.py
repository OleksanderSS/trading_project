#!/usr/bin/env python3
"""
Реальний збирач data з API для trading системи
Підтримка Yahoo Finance, Alpha Vantage, та інших джерел
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config.trading_config import TradingConfig
from utils.enhanced_data_validator import DataValidator
from utils.common_utils import CacheManager, PerformanceMonitor


class RealDataCollector:
    """Збирач реальних ринкових data"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        
        # API keys з конфігурації
        self.alpha_vantage_key = getattr(config, 'alpha_vantage_key', None)
        self.finnhub_key = getattr(config, 'finnhub_key', None)
        
        # Rate limiting
        self.rate_limiter = {
            'yahoo': {'calls': 0, 'last_reset': time.time(), 'limit': 2000, 'window': 3600},
            'alpha_vantage': {'calls': 0, 'last_reset': time.time(), 'limit': 5, 'window': 60},
            'finnhub': {'calls': 0, 'last_reset': time.time(), 'limit': 60, 'window': 60}
        }
        
        self._lock = threading.Lock()
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Перевірка rate limiting"""
        with self._lock:
            limiter = self.rate_limiter[api_name]
            current_time = time.time()
            
            # Скидання лічильника якщо вікно закінчилося
            if current_time - limiter['last_reset'] > limiter['window']:
                limiter['calls'] = 0
                limiter['last_reset'] = current_time
            
            if limiter['calls'] >= limiter['limit']:
                wait_time = limiter['window'] - (current_time - limiter['last_reset'])
                self.logger.warning(f"Rate limit reached for {api_name}. Waiting {wait_time:.2f}s")
                time.sleep(wait_time + 1)
                limiter['calls'] = 0
                limiter['last_reset'] = time.time()
                return False
            
            limiter['calls'] += 1
            return True
    
    def get_yahoo_finance_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Отримання data з Yahoo Finance"""
        return self.performance_monitor.time_execution("yahoo_finance_data")(self._get_yahoo_finance_data_impl)(ticker, period, interval)
    
    def _get_yahoo_finance_data_impl(self, ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Отримання data з Yahoo Finance"""
        if not self._check_rate_limit('yahoo'):
            return None
        
        try:
            # Перевірка кешу
            cache_key = f"yahoo_{ticker}_{period}_{interval}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached data for {ticker}")
                return cached_data
            
            # Завантаження data
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data is None or data.empty:
                self.logger.warning(f"No data found for {ticker}")
                return None
            
            # Стандартизація колонок
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data.reset_index(inplace=True)
            
            # Перевірка наявності колонки 'date' після reset_index
            if 'date' not in data.columns and 'datetime' in data.columns:
                data.rename(columns={'datetime': 'date'}, inplace=True)
            elif 'date' not in data.columns and 'index' in data.columns:
                data.rename(columns={'index': 'date'}, inplace=True)
            
            # Перейменування колонок
            column_mapping = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)
            
            # Валідація data
            try:
                # Перевірка наявності необхідних колонок
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = set(required_columns) - set(data.columns)
                if missing_cols:
                    self.logger.warning(f"Missing required columns for {ticker}: {missing_cols}")
                    return None
                
                # Проста валідація without Pydantic
                if data.empty:
                    self.logger.warning(f"No data found for {ticker}")
                    return None
                    
            except Exception as e:
                self.logger.warning(f"Data validation failed for {ticker}: {e}")
                return None
            
            # Кешування результату
            self.cache_manager.set(cache_key, data, ttl=3600)  # 1 година
            
            self.logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None
    
    def get_alpha_vantage_data(self, ticker: str, function: str = "TIME_SERIES_DAILY") -> Optional[pd.DataFrame]:
        """Отримання data з Alpha Vantage"""
        return self.performance_monitor.time_execution("alpha_vantage_data")(self._get_alpha_vantage_data_impl)(ticker, function)
    
    def _get_alpha_vantage_data_impl(self, ticker: str, function: str = "TIME_SERIES_DAILY") -> Optional[pd.DataFrame]:
        """Отримання data з Alpha Vantage"""
        if not self.alpha_vantage_key:
            self.logger.warning("Alpha Vantage API key not configured")
            return None
        
        if not self._check_rate_limit('alpha_vantage'):
            return None
        
        try:
            # Перевірка кешу
            cache_key = f"av_{ticker}_{function}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
            
            # API запит
            url = "https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': ticker,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Перевірка помилок
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None
            
            # Парсинг data
            time_series_key = f"Time Series ({function.split('_')[-1]})"
            if time_series_key not in data:
                self.logger.error(f"Unexpected response format for {ticker}")
                return None
            
            time_series = data[time_series_key]
            
            # Конвертація в DataFrame
            records = []
            for date_str, values in time_series.items():
                record = {
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Кешування результату
            self.cache_manager.set(cache_key, df, ttl=3600)
            
            self.logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Alpha Vantage data for {ticker}: {e}")
            return None
    
    def get_finnhub_data(self, ticker: str, resolution: str = "D", count: int = 365) -> Optional[pd.DataFrame]:
        """Отримання data з Finnhub"""
        return self.performance_monitor.time_execution("finnhub_data")(self._get_finnhub_data_impl)(ticker, resolution, count)
    
    def _get_finnhub_data_impl(self, ticker: str, resolution: str = "D", count: int = 365) -> Optional[pd.DataFrame]:
        """Отримання data з Finnhub"""
        if not self.finnhub_key:
            self.logger.warning("Finnhub API key not configured")
            return None
        
        if not self._check_rate_limit('finnhub'):
            return None
        
        try:
            # Перевірка кешу
            cache_key = f"finnhub_{ticker}_{resolution}_{count}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
            
            # API запит
            end_date = int(time.time())
            start_date = end_date - (count * 24 * 60 * 60)  # count днів назад
            
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': ticker,
                'resolution': resolution,
                'from': start_date,
                'to': end_date,
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Перевірка помилок
            if data.get('s') != 'ok':
                self.logger.error(f"Finnhub error: {data}")
                return None
            
            # Конвертація в DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            
            # Кешування результату
            self.cache_manager.set(cache_key, df, ttl=3600)
            
            self.logger.info(f"Successfully fetched {len(df)} records from Finnhub for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Finnhub data for {ticker}: {e}")
            return None
    
    def get_real_time_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Отримання реальних data (поточна ціна)"""
        try:
            # Спроба Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and 'regularMarketPrice' in info:
                return {
                    'ticker': ticker,
                    'price': info['regularMarketPrice'],
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'timestamp': datetime.now(),
                    'source': 'yahoo_finance'
                }
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance real-time data failed for {ticker}: {e}")
        
        # Fallback до Alpha Vantage
        if self.alpha_vantage_key:
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': ticker,
                    'apikey': self.alpha_vantage_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    return {
                        'ticker': ticker,
                        'price': float(quote['05. price']),
                        'change': float(quote['09. change']),
                        'change_percent': float(quote['10. change percent'].replace('%', '')),
                        'volume': int(quote['06. volume']),
                        'timestamp': datetime.now(),
                        'source': 'alpha_vantage'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Alpha Vantage real-time data failed for {ticker}: {e}")
        
        return None
    
    def collect_batch_data(self, tickers: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Масове збирання data з паралельною обробкою"""
        return self.performance_monitor.time_execution("batch_data_collection")(self._collect_batch_data_impl)(tickers, period, interval)
    
    def _collect_batch_data_impl(self, tickers: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Масове збирання data з паралельною обробкою"""
        results = {}
        
        def fetch_ticker_data(ticker):
            """Функція для отримання data одного тікера"""
            try:
                # Спроба Yahoo Finance першою
                data = self.get_yahoo_finance_data(ticker, period, interval)
                if data is not None:
                    return ticker, data
                
                # Fallback до Alpha Vantage
                if self.alpha_vantage_key:
                    data = self.get_alpha_vantage_data(ticker)
                    if data is not None:
                        return ticker, data
                
                # Fallback до Finnhub
                if self.finnhub_key:
                    data = self.get_finnhub_data(ticker)
                    if data is not None:
                        return ticker, data
                
                return ticker, None
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {e}")
                return ticker, None
        
        # Паралельна обробка
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(fetch_ticker_data, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker, data = future.result()
                if data is not None:
                    results[ticker] = data
                else:
                    self.logger.warning(f"Failed to fetch data for {ticker}")
        
        self.logger.info(f"Successfully collected data for {len(results)}/{len(tickers)} tickers")
        return results
    
    def get_market_indicators(self) -> Dict[str, Any]:
        """Отримання ринкових індикаторів"""
        indicators = {}
        
        # Основі індекси
        market_indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
        for index in market_indices:
            try:
                data = self.get_real_time_data(index)
                if data:
                    indicators[index] = data
            except Exception as e:
                self.logger.warning(f"Failed to get data for {index}: {e}")
        
        return indicators
    
    def save_data_to_file(self, data: Dict[str, pd.DataFrame], filename: str = None) -> str:
        """Збереження data у файл"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_data_{timestamp}.json"
        
        filepath = self.config.data.data_dir / 'real_data' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Конвертація DataFrame у JSON-сумісний формат
        json_data = {}
        for ticker, df in data.items():
            json_data[ticker] = df.to_dict('records')
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': json_data,
                'source': 'real_data_collector'
            }, f, indent=2, default=str)
        
        self.logger.info(f"Data saved to {filepath}")
        return str(filepath)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Отримання звіту про продуктивність"""
        return self.performance_monitor.get_performance_report()
