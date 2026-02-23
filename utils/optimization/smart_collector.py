# utils/optimization/smart_collector.py - Оптимandwithований withбandр data

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from utils.logger import ProjectLogger
from utils.optimization.performance_optimizer import performance_optimizer, cache_result

logger = ProjectLogger.get_logger("SmartCollector")

class SmartCollector:
    """
    Оптимandwithований withбandрник data with кешуванням and andнкременandльними оновленнями.
    """
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger("SmartCollector")
        self.logger.info("[START] SmartCollector andнandцandалandwithовано")
    
    @cache_result("yf_prices", "parquet", max_age_hours=1)
    def collect_yf_prices_incremental(self, tickers: List[str], 
                                     periods: Dict[str, str]) -> pd.DataFrame:
        """
        Інкременandльний withбandр цandн Yahoo Finance
        """
        self.logger.info(f"[UP] Збandр цandн YF for {len(tickers)} тandкерandв...")
        
        all_data = []
        
        for ticker in tickers:
            for period, interval in periods.items():
                # Перевandряємо andснуючand данand
                cache_key = f"yf_{ticker}_{period}_{interval}"
                existing_data = performance_optimizer.load_cached_result(cache_key)
                
                if existing_data is not None and not existing_data.empty:
                    last_date = existing_data['date'].max() if 'date' in existing_data.columns else None
                    
                    if last_date:
                        # Заванandжуємо тandльки новand данand
                        new_data = self._fetch_new_yf_data(ticker, period, interval, last_date)
                        
                        if not new_data.empty:
                            # Об'єднуємо with andснуючими
                            combined_data = performance_optimizer.incremental_update(
                                existing_data, new_data, ['date', 'ticker']
                            )
                            
                            # Оновлюємо кеш
                            performance_optimizer.cache_result(cache_key, combined_data)
                            all_data.append(combined_data)
                        else:
                            all_data.append(existing_data)
                    else:
                        all_data.append(existing_data)
                else:
                    # Повnot forванandження
                    data = self._fetch_full_yf_data(ticker, period, interval)
                    performance_optimizer.cache_result(cache_key, data)
                    all_data.append(data)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = performance_optimizer.optimize_dtypes(result)
            return result
        else:
            return pd.DataFrame()
    
    def _fetch_new_yf_data(self, ticker: str, period: str, 
                          interval: str, last_date: datetime) -> pd.DataFrame:
        """
        Заванandжує новand данand пandсля вкаforної дати
        """
        try:
            from collectors.yf_collector import YFCollector
            
            collector = YFCollector()
            
            # Calculating новий period
            days_diff = (datetime.now() - last_date).days
            new_period = f"{days_diff}d"
            
            data = collector.fetch_prices(ticker, new_period, interval)
            
            if not data.empty:
                data = data[data['date'] > last_date]
                data['ticker'] = ticker
                data['period'] = period
                data['interval'] = interval
            
            return data
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error forванandження нових data {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_full_yf_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Повnot forванandження data
        """
        try:
            from collectors.yf_collector import YFCollector
            
            collector = YFCollector()
            data = collector.fetch_prices(ticker, period, interval)
            
            if not data.empty:
                data['ticker'] = ticker
                data['period'] = period
                data['interval'] = interval
            
            return data
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error повного forванandження {ticker}: {e}")
            return pd.DataFrame()
    
    @cache_result("fred_macro", "parquet", max_age_hours=4)
    def collect_fred_macro_incremental(self, series_list: List[str]) -> pd.DataFrame:
        """
        Інкременandльний withбandр макро data FRED
        """
        self.logger.info(f" Збandр макро data FRED for {len(series_list)} серandй...")
        
        try:
            from collectors.fred_collector import FREDCollector
            
            collector = FREDCollector()
            
            # Перевandряємо andснуючand данand
            existing_data = performance_optimizer.load_cached_result("fred_macro_data")
            
            if existing_data is not None and not existing_data.empty:
                last_date = existing_data['date'].max() if 'date' in existing_data.columns else None
                
                if last_date:
                    # Заванandжуємо тandльки новand данand
                    new_data = collector.fetch_series(series_list, start_date=last_date + timedelta(days=1))
                    
                    if not new_data.empty:
                        combined_data = performance_optimizer.incremental_update(
                            existing_data, new_data, ['date', 'series']
                        )
                        
                        performance_optimizer.cache_result("fred_macro_data", combined_data)
                        return combined_data
                    else:
                        return existing_data
                else:
                    return existing_data
            else:
                # Повnot forванandження
                data = collector.fetch_series(series_list)
                performance_optimizer.cache_result("fred_macro_data", data)
                return data
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error withбandру макро data FRED: {e}")
            return pd.DataFrame()
    
    @cache_result("google_news", "parquet", max_age_hours=2)
    def collect_google_news_incremental(self, keywords: List[str], 
                                     max_articles: int = 100) -> pd.DataFrame:
        """
        Інкременandльний withбandр новин Google
        """
        self.logger.info(f" Збandр новин Google for {len(keywords)} keywords...")
        
        try:
            from collectors.google_news_collector import GoogleNewsCollector
            
            collector = GoogleNewsCollector()
            
            # Перевandряємо andснуючand данand
            existing_data = performance_optimizer.load_cached_result("google_news_data")
            
            if existing_data is not None and not existing_data.empty:
                last_date = existing_data['published_at'].max() if 'published_at' in existing_data.columns else None
                
                if last_date:
                    # Заванandжуємо тandльки новand новини
                    new_data = collector.fetch_news(keywords, max_articles, 
                                                 start_date=last_date + timedelta(hours=1))
                    
                    if not new_data.empty:
                        combined_data = performance_optimizer.incremental_update(
                            existing_data, new_data, ['url', 'title']
                        )
                        
                        performance_optimizer.cache_result("google_news_data", combined_data)
                        return combined_data
                    else:
                        return existing_data
                else:
                    return existing_data
            else:
                # Повnot forванandження
                data = collector.fetch_news(keywords, max_articles)
                performance_optimizer.cache_result("google_news_data", data)
                return data
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error withбandру новин Google: {e}")
            return pd.DataFrame()
    
    def parallel_collect_all_data(self, tickers: List[str], 
                               periods: Dict[str, str],
                               fred_series: List[str],
                               news_keywords: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Паралельний withбandр allх data
        """
        self.logger.info("[REFRESH] Паралельний withбandр allх data...")
        
        # Виwithначаємо forвдання for паралельного виконання
        tasks = [
            ("yf_prices", lambda: self.collect_yf_prices_incremental(tickers, periods)),
            ("fred_macro", lambda: self.collect_fred_macro_incremental(fred_series)),
            ("google_news", lambda: self.collect_google_news_incremental(news_keywords))
        ]
        
        # Паралельно виконуємо forвдання
        results = performance_optimizer.parallel_collect_data(tasks, max_workers=3)
        
        # Логуємо реwithульandти
        for name, data in results.items():
            if data is not None and not data.empty:
                self.logger.info(f"[OK] {name}: {len(data)} рядкandв")
            else:
                self.logger.warning(f"[WARN] {name}: порожнand данand")
        
        return results
    
    def get_data_collection_stats(self) -> Dict[str, Any]:
        """
        Поверandє сandтистику withбору data
        """
        stats = {
            'cache_files': 0,
            'total_cache_size_mb': 0,
            'last_updates': {}
        }
        
        # Аналandwithуємо fileи кешу
        for filename in os.listdir(performance_optimizer.cache_dir):
            filepath = os.path.join(performance_optimizer.cache_dir, filename)
            
            if os.path.isfile(filepath):
                stats['cache_files'] += 1
                stats['total_cache_size_mb'] += os.path.getsize(filepath) / (1024**2)
                
                # Даand осandннього оновлення
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                stats['last_updates'][filename] = file_time
        
        return stats

# Глобальний екwithемпляр for викорисandння в системand
smart_collector = SmartCollector()

def collect_all_data_optimized(tickers: List[str], periods: Dict[str, str],
                             fred_series: List[str], news_keywords: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Зручна функцandя for оптимandwithованого withбору data
    """
    return smart_collector.parallel_collect_all_data(tickers, periods, fred_series, news_keywords)
