"""
FREE Google Trends Collector - Безкоштовний колектор Google Trends
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pytrends.request import TrendReq

from .collector_interface import CollectorStatus, CollectorType, CollectionResult
from .base_collector import BaseCollector

logger = logging.getLogger("FreeGoogleTrendsCollector")


class FreeGoogleTrendsCollector(BaseCollector):
    """
    Безкоштовний колектор Google Trends
    
    Обмеження:
    - Обмеження API (withoutкоштовна версія)
    - Кешування обов'язкове
    - Повільні запити
    """
    
    def __init__(
        self,
        geo: str = 'US',
        timeframe: str = 'today 3-m',
        hl: str = 'en-US',
        tz: int = 360,
        **kwargs
    ):
        # Ініціалізація базового колектора
        self.geo = geo
        self.timeframe = timeframe
        self.hl = hl
        self.tz = tz
        
        # Ініціалізація pytrends
        try:
            self.pytrends = TrendReq(hl=hl, tz=tz)
            logger.info(f"[OK] FreeGoogleTrendsCollector initialized: geo={geo}, timeframe={timeframe}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize pytrends: {e}")
            raise ConfigurationError(f"Failed to initialize pytrends: {e}")
        
        # Ініціалізуємо cache_manager - використовуємо IntelligentCacheManager
        try:
            from core.stages.stage_1_collectors_layer import IntelligentCacheManager
            self.cache_manager = IntelligentCacheManager(
                cache_dir="cache/google_trends",
                max_memory_gb=1.0,
                cache_ttl_hours=6
            )
        except Exception as e:
            logger.warning(f"Could not initialize IntelligentCacheManager: {e}")
            # Fallback до простого кешування
            self.cache_manager = None
    
    def collect_data(self, keywords: List[str], **kwargs) -> pd.DataFrame:
        """
        Основний метод збору data для сумісності з іншими колекторами
        
        Args:
            keywords: Список ключових слів
            **kwargs: Додаткові параметри
            
        Returns:
            pd.DataFrame: Зібрані дані
        """
        result = self.collect(keywords, **kwargs)
        
        if result.success and result.data:
            # Конвертуємо CollectionResult в DataFrame
            trends_data = result.data
            
            # Створюємо DataFrame
            df_data = []
            for keyword, data in trends_data.items():
                if isinstance(data, dict) and 'timeline' in data:
                    timeline = data['timeline']
                    for date_str, value in timeline.items():
                        df_data.append({
                            'keyword': keyword,
                            'date': date_str,
                            'value': value,
                            'geo': self.geo,
                            'timeframe': self.timeframe
                        })
            
            return pd.DataFrame(df_data)
        else:
            return pd.DataFrame()
    
    def collect(self, tickers: List[str] = None, **kwargs) -> CollectionResult:
        """
        Збирає Google Trends для тікерів або ключових слів зі словника
        
        Args:
            tickers: Список тікерів (якщо None, використовує словник)
            
        Returns:
            CollectionResult: Результат збору
        """
        start_time = time.time()
        start_dt = datetime.now()
        
        try:
            # Якщо тікери не надані, використовує словник ключових слів
            if tickers is None:
                from config.config_loader import load_yaml_config
                import os
                base_dir = os.path.dirname(__file__)
                config_path = os.path.join(base_dir, "..", "config", "news_sources.yaml")
                config = load_yaml_config(config_path)
                keyword_dict = config.get("keywords", {})
                
                # ВИПРАВЛЕНО: виключаємо категорію tickers
                keyword_dict = {k: v for k, v in keyword_dict.items() if k != "tickers"}
                
                # Збираємо всі ключові слова
                keywords = []
                for category, words in keyword_dict.items():
                    keywords.extend(words)
                
                # ВИПРАВЛЕНО: стабільний порядок для кешу
                keywords = sorted(set(keywords))
                keywords = keywords[:10]
                tickers = keywords
                logger.info(f"[GoogleTrends] Using {len(tickers)} keywords from dictionary (tickers excluded)")
            
            # ВИПРАВЛЕНО: стабільний cache key
            cache_key = f"google_trends_{'_'.join(sorted(tickers))}_{self.timeframe}_{self.geo}"
            if self.cache_manager:
                try:
                    cached_data = self.cache_manager.get("google_trends", cache_key)
                    if cached_data is not None:
                        end_dt = datetime.now()
                        logger.info(f"[OK] Using cached Google Trends data for {len(tickers)} keywords")
                        return CollectionResult(
                            data=cached_data,
                            status=CollectorStatus.COMPLETED,
                            message="Using cached data",
                            metadata={
                                'source': 'google_trends_cached',
                                'keywords': tickers,
                                'collection_time': time.time() - start_time
                            },
                            start_time=start_dt,
                            end_time=end_dt,
                            records_count=len(cached_data) if hasattr(cached_data, '__len__') else 0,
                            errors=[]
                        )
                except Exception as cache_error:
                    logger.warning(f"Cache error: {cache_error}")
            else:
                logger.warning("No cache manager available")
            
            # Збираємо дані
            trends_data = self._collect_trends_data(tickers)
            
            # Кешуємо результат
            if self.cache_manager:
                try:
                    self.cache_manager.set("google_trends", cache_key, trends_data, expire_hours=6)  # 6 годин кеш
                except Exception as cache_error:
                    logger.warning(f"Cache set error: {cache_error}")
            
            end_dt = datetime.now()
            return CollectionResult(
                data=trends_data,
                status=CollectorStatus.COMPLETED,
                message="Data collected successfully",
                metadata={
                    'source': 'google_trends',
                    'keywords': tickers,
                    'geo': self.geo,
                    'timeframe': self.timeframe,
                    'collection_time': time.time() - start_time
                },
                start_time=start_dt,
                end_time=end_dt,
                records_count=len(trends_data) if hasattr(trends_data, '__len__') else 0,
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"[ERROR] Google Trends collection failed: {e}")
            end_dt = datetime.now()
            return CollectionResult(
                data=pd.DataFrame(),
                status=CollectorStatus.ERROR,
                message=f"Collection failed: {e}",
                metadata={'collection_time': time.time() - start_time, 'error': str(e)},
                start_time=start_dt,
                end_time=end_dt,
                records_count=0,
                errors=[str(e)]
            )
    
    def _collect_trends_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Збирає дані трендів для тікерів
        
        Args:
            tickers: Список тікерів
            
        Returns:
            Dict: Дані трендів
        """
        trends_data = {}
        
        # Обробляємо тікери групами (обмеження API)
        batch_size = 5  # Максимальна група для pytrends
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            # Retry логіка для 429 помилок
            max_retries = 3
            retry_delay = 5  # секунд
            
            for retry in range(max_retries):
                try:
                    # Запит до Google Trends
                    self.pytrends.build_payload(
                        kw_list=batch,
                        timeframe=self.timeframe,
                        geo=self.geo
                    )
                    
                    # Отримуємо дані
                    interest_data = self.pytrends.interest_over_time()
                    
                    if not interest_data.empty:
                        # Обробляємо дані для кожного тікера
                        for ticker in batch:
                            if ticker in interest_data.columns:
                                ticker_data = interest_data[ticker]
                                
                                trends_data[ticker] = {
                                    'trend_score': ticker_data.iloc[-1] if len(ticker_data) > 0 else 0,
                                    'trend_mean': ticker_data.mean(),
                                    'trend_std': ticker_data.std(),
                                    'trend_max': ticker_data.max(),
                                    'trend_min': ticker_data.min(),
                                    'trend_change': self._calculate_trend_change(ticker_data),
                                    'trend_volatility': ticker_data.std() / ticker_data.mean() if ticker_data.mean() > 0 else 0,
                                    'data_points': len(ticker_data),
                                    'last_updated': datetime.now().isoformat()
                                }
                            else:
                                # Якщо data немає
                                trends_data[ticker] = {
                                    'trend_score': 0,
                                    'trend_mean': 0,
                                    'trend_std': 0,
                                    'trend_max': 0,
                                    'trend_min': 0,
                                    'trend_change': 0,
                                    'trend_volatility': 0,
                                    'data_points': 0,
                                    'last_updated': datetime.now().isoformat()
                                }
                    
                    # Успішний запит - виходимо з retry циклу
                    break
                    
                except Exception as e:
                    if "429" in str(e) and retry < max_retries - 1:
                        logger.warning(f"[WARN] Rate limited for batch {batch}, retry {retry + 1}/{max_retries} after {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Експоненційна затримка
                        continue
                    elif "429" in str(e):
                        logger.error(f"[WARN] Failed to collect trends for batch {batch}: The request failed: Google returned a response with code 429")
                        break  # Виходимо з циклу retry при постійному rate limiting
                    else:
                        logger.warning(f"[WARN] Error collecting trends for batch {batch}: {e}")
                        break
            
            # Затримка між запитами (збільшено для уникнення 429)
            time.sleep(3)  # Збільшено з 1 до 3 секунд
        
        return trends_data
    
    def _calculate_trend_change(self, trend_series: pd.Series) -> float:
        """
        Розраховує зміну тренду
        
        Args:
            trend_series: Серія data тренду
            
        Returns:
            float: Зміна тренду у відсотках
        """
        if len(trend_series) < 2:
            return 0.0
        
        first_value = trend_series.iloc[0]
        last_value = trend_series.iloc[-1]
        
        if first_value == 0:
            return 0.0
        
        return (last_value - first_value) / first_value * 100
    
    def get_collector_info(self) -> Dict:
        """
        Повертає інформацію про колектор
        """
        return {
            'name': 'FreeGoogleTrendsCollector',
            'type': CollectorType.ALTERNATIVE_DATA,
            'description': 'Безкоштовний колектор Google Trends',
            'source': 'Google Trends API',
            'cost': 'Free',
            'limitations': [
                'Rate limited',
                'Limited to 5 tickers per request',
                'Data delay: 1-3 days',
                'Geographic restrictions'
            ],
            'update_frequency': 'Daily',
            'data_retention': '6 hours cache',
            'supported_geos': ['US', 'GB', 'DE', 'FR', 'JP', 'CA', 'AU'],
            'default_timeframe': 'today 3-m'
        }
