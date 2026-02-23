#!/usr/bin/env python3
"""
Real Data Collector - Використовує існуючі етапи pipeline для збору реальних data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time

# Використовуємо існуючі етапи pipeline
# ВИПРАВЛЕНО: видаляємо імпорт видаленої функції
# from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrichment

logger = logging.getLogger(__name__)


class RealDataCollector:
    """
    Колектор реальних data для паперової торгівлі
    Використовує існуючі етапи pipeline для збору data
    """
    
    def __init__(self):
        # Розширений список тікерів для реального трейдингу
        self.tickers = [
            # Large Cap ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
            
            # Tech Stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'COST',
            
            # Industrial
            'BA', 'CAT', 'GE', 'MMM', 'HON',
            
            # Materials
            'LIN', 'BHP', 'RIO', 'DOW',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'AEP',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX'
        ]
        
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        
        # Кеш для оптимізації
        self.data_cache = {}
        self.last_update = {}
        
        logger.info(f"[DATA] Real Data Collector initialized for {len(self.tickers)} tickers")
    
    def get_real_time_data(self, tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Отримання реальних data в реальному часі
        Використовує існуючий Stage 1 pipeline
        """
        if tickers is None:
            tickers = self.tickers
        
        logger.info(f"[DATA] Getting real-time data using Stage 1 pipeline...")
        
        try:
            # ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
            from core.stages.stage_manager import StageManager
            stage_manager = StageManager()
            stage1_data = stage_manager.run_stage_1(force_refresh=False)
            
            # Фільтруємо дані для потрібних тікерів
            filtered_data = {}
            
            # Обробляємо цінові дані
            if 'prices' in stage1_data and not stage1_data['prices'].empty:
                price_df = stage1_data['prices']
                
                for ticker in tickers:
                    if 'ticker' in price_df.columns:
                        ticker_data = price_df[price_df['ticker'] == ticker].copy()
                        if not ticker_data.empty:
                            # Додаємо технічні індикатори
                            ticker_data = self._add_technical_indicators(ticker_data)
                            filtered_data[ticker] = ticker_data
                            logger.debug(f"[OK] {ticker}: {len(ticker_data)} rows")
            
            logger.info(f"[DATA] Real-time data collected: {len(filtered_data)} tickers")
            return filtered_data
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting real-time data: {e}")
            return {}
    
    def get_comprehensive_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Отримання комплексних ринкових data
        Використовує існуючі Stage 1 та Stage 2
        """
        logger.info("[RESTART] Getting comprehensive market data using existing pipeline...")
        
        try:
            # ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
            from core.stages.stage_manager import StageManager
            stage_manager = StageManager()
            stage1_data = stage_manager.run_stage_1(force_refresh=False)
            logger.info("[OK] Stage 1 data collection completed")
            
            # Stage 2: Збагачення data
            merged_df, _, _, _ = run_stage_2_enrichment(stage1_data)
            logger.info("[OK] Stage 2 data enrichment completed")
            
            # Структуруємо дані для зручності
            comprehensive_data = {
                'stage1_data': stage1_data,
                'enriched_data': merged_df,
                'price_data': self._extract_price_data(stage1_data),
                'news_data': self._extract_news_data(stage1_data),
                'macro_data': self._extract_macro_data(stage1_data),
                'indices_data': self._extract_indices_data(stage1_data)
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting comprehensive market data: {e}")
            return {}
    
    def _extract_price_data(self, stage1_data: Dict) -> Dict[str, pd.DataFrame]:
        """Вилучення цінових data з Stage 1"""
        price_data = {}
        
        if 'prices' in stage1_data and not stage1_data['prices'].empty:
            price_df = stage1_data['prices']
            
            for ticker in self.tickers:
                if 'ticker' in price_df.columns:
                    ticker_data = price_df[price_df['ticker'] == ticker].copy()
                    if not ticker_data.empty:
                        price_data[ticker] = ticker_data
        
        return price_data
    
    def _extract_news_data(self, stage1_data: Dict) -> pd.DataFrame:
        """Вилучення новинних data з Stage 1"""
        if 'all_news' in stage1_data:
            return stage1_data['all_news']
        return pd.DataFrame()
    
    def _extract_macro_data(self, stage1_data: Dict) -> pd.DataFrame:
        """Вилучення макро data з Stage 1"""
        if 'macro' in stage1_data:
            return stage1_data['macro']
        return pd.DataFrame()
    
    def _extract_indices_data(self, stage1_data: Dict) -> pd.DataFrame:
        """Вилучення data індексів з Stage 1"""
        indices_data = {}
        
        # Спробуємо знайти індекси в цінових data
        if 'prices' in stage1_data and not stage1_data['prices'].empty:
            price_df = stage1_data['prices']
            indices = ['^VIX', '^DXY', '^TNX', '^TYX', '^IRX']
            
            for index in indices:
                if 'ticker' in price_df.columns:
                    index_data = price_df[price_df['ticker'] == index].copy()
                    if not index_data.empty:
                        indices_data[index] = index_data
        
        return pd.DataFrame(indices_data) if indices_data else pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Додавання технічних індикаторів до data
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        # RSI
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) >= 26:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if len(df) >= 20:
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Moving Averages
        if len(df) >= 50:
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['ma_200'] = df['close'].rolling(window=200).mean()
        
        # Volume indicators
        if 'volume' in df.columns and len(df) >= 20:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = abs(df['price_change'])
        
        return df
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Отримання поточної ціни для тікера
        """
        try:
            # Використовуємо існуючий YFCollector
            from collectors.yf_collector import YFCollector
            yf = YFCollector()
            
            # Отримуємо останні дані
            data = yf.fetch(ticker=ticker, interval='1m', period='1d')
            
            if not data.empty and 'close' in data.columns:
                return float(data['close'].iloc[-1])
            else:
                return None
                
        except Exception as e:
            logger.error(f"[ERROR] Error getting price for {ticker}: {e}")
            return None
    
    def get_market_sentiment(self) -> Dict[str, float]:
        """
        Отримання ринкового сентименту
        """
        try:
            # ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
            from core.stages.stage_manager import StageManager
            stage_manager = StageManager()
            stage1_data = stage_manager.run_stage_1(force_refresh=False)
            
            # VIX як індекс страху
            vix = self.get_current_price("^VIX")
            vix_sentiment = 1 - (vix - 10) / 40  # Нормалізуємо VIX до 0-1
            vix_sentiment = max(0, min(1, vix_sentiment))  # Обмежуємо між 0-1
            
            # S&P 500 зміна
            spy_change = self._get_price_change("SPY")
            
            # News sentiment (якщо доступно)
            news_sentiment = 0.5  # Нейтральний за замовчуванням
            if 'all_news' in stage1_data and not stage1_data['all_news'].empty:
                news_df = stage1_data['all_news']
                if 'sentiment_score' in news_df.columns:
                    news_sentiment = news_df['sentiment_score'].mean()
            
            return {
                'vix_sentiment': vix_sentiment,
                'spy_change': spy_change,
                'news_sentiment': news_sentiment,
                'overall_sentiment': (vix_sentiment + (1 if spy_change > 0 else 0) + news_sentiment) / 3
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting market sentiment: {e}")
            return {'overall_sentiment': 0.5}
    
    def _get_price_change(self, ticker: str) -> float:
        """
        Отримання зміни ціни для тікера
        """
        try:
            # Використовуємо існуючий YFCollector
            from collectors.yf_collector import YFCollector
            yf = YFCollector()
            
            data = yf.fetch(ticker=ticker, interval='1d', period='2d')
            
            if len(data) >= 2 and 'close' in data.columns:
                yesterday_close = data['close'].iloc[-2]
                today_close = data['close'].iloc[-1]
                change_pct = (today_close - yesterday_close) / yesterday_close
                return change_pct
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"[ERROR] Error getting price change for {ticker}: {e}")
            return 0.0
    
    def update_data_cache(self, tickers: List[str] = None):
        """
        Оновлення кешу data
        """
        if tickers is None:
            tickers = self.tickers
        
        for ticker in tickers:
            try:
                # Оновлюємо дані кожні 5 хвилин
                if ticker not in self.last_update or \
                   datetime.now() - self.last_update[ticker] > timedelta(minutes=5):
                    
                    # Використовуємо існуючий pipeline
                    data = self.get_historical_data(ticker, period="5d", interval="15m")
                    if not data.empty:
                        self.data_cache[ticker] = data
                        self.last_update[ticker] = datetime.now()
                        logger.debug(f"[OK] Updated cache for {ticker}")
                        
            except Exception as e:
                logger.error(f"[ERROR] Error updating cache for {ticker}: {e}")
                continue
    
    def get_historical_data(self, ticker: str, period: str = "1mo", 
                          interval: str = "1h") -> pd.DataFrame:
        """
        Отримання історичних data для тікера
        """
        try:
            # Використовуємо існуючий YFCollector
            from collectors.yf_collector import YFCollector
            yf = YFCollector()
            
            data = yf.fetch(ticker=ticker, interval=interval, period=period)
            
            if not data.empty:
                data = self._add_technical_indicators(data)
                logger.info(f"[OK] {ticker}: {len(data)} rows of historical data")
                return data
            else:
                logger.warning(f"[WARN] {ticker}: No historical data available")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[ERROR] Error getting historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_cached_data(self, ticker: str) -> pd.DataFrame:
        """
        Отримання data з кешу
        """
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        else:
            # Якщо data немає в кеші, отримуємо їх
            data = self.get_historical_data(ticker, period="5d", interval="15m")
            if not data.empty:
                self.data_cache[ticker] = data
                self.last_update[ticker] = datetime.now()
            return data
