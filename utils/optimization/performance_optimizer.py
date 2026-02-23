# utils/optimization/performance_optimizer.py - Оптимandforцandя ресурсandв for трейдингу

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import psutil
import gc
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("PerformanceOptimizer")

class PerformanceOptimizer:
    """
    Оптимandforтор продуктивностand for трейдингової system.
    Кешування, паралельна обробка, оптимandforцandя пам'ятand.
    """
    
    def __init__(self, cache_dir: str = "data/cache", max_cache_age_hours: int = 4):
        self.cache_dir = cache_dir
        self.max_cache_age_hours = max_cache_age_hours
        self.logger = ProjectLogger.get_logger("PerformanceOptimizer")
        
        # Створюємо директорandю кешу
        os.makedirs(cache_dir, exist_ok=True)
        
        # [NEW] Монandторинг ресурсandв
        self.resource_monitor = ResourceMonitor()
        
        self.logger.info(f"[START] PerformanceOptimizer andнandцandалandwithовано")
        self.logger.info(f"  - Директорandя кешу: {cache_dir}")
        self.logger.info(f"  - Максимальний вandк кешу: {max_cache_age_hours} годин")
    
    def cache_result(self, cache_key: str, data: Any, file_format: str = 'parquet'):
        """
        Кешує реwithульandт у file
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.{file_format}")
        
        try:
            if file_format == 'parquet':
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(cache_file, index=False)
                else:
                    pd.DataFrame(data).to_parquet(cache_file, index=False)
            elif file_format == 'feather':
                if isinstance(data, pd.DataFrame):
                    data.to_feather(cache_file)
                else:
                    pd.DataFrame(data).to_feather(cache_file)
            elif file_format == 'pickle':
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            
            self.logger.debug(f" Закешовано: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"[WARN] Error кешування {cache_key}: {e}")
    
    def load_cached_result(self, cache_key: str, file_format: str = 'parquet', 
                         max_age_hours: Optional[int] = None) -> Optional[Any]:
        """
        Заванandжує реwithульandт with кешу
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.{file_format}")
        
        if not os.path.exists(cache_file):
            return None
        
        # Перевandряємо вandк кешу
        max_age = max_age_hours or self.max_cache_age_hours
        file_age = time.time() - os.path.getmtime(cache_file)
        
        if file_age > max_age * 3600:
            self.logger.debug(f" Кеш forсandрandв: {cache_key}")
            return None
        
        try:
            if file_format == 'parquet':
                data = pd.read_parquet(cache_file)
            elif file_format == 'feather':
                data = pd.read_feather(cache_file)
            elif file_format == 'pickle':
                import pickle
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            
            self.logger.debug(f" Заванandжено with кешу: {cache_key}")
            return data
            
        except Exception as e:
            self.logger.warning(f"[WARN] Error forванandження кешу {cache_key}: {e}")
            return None
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Оптимandwithує типи data for економandї пам'ятand
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        df_optimized = df.copy()
        
        # [NEW] Оптимandforцandя числових типandв
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:  # Signed integers
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # [NEW] Оптимandforцandя float типandв
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # [NEW] Оптимandforцandя категорandальних типandв
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Менше 50% унandкальних
                df_optimized[col] = df_optimized[col].astype('category')
        
        # [NEW] Оптимandforцandя datetime
        for col in df_optimized.select_dtypes(include=['datetime64[ns]']).columns:
            df_optimized[col] = pd.to_datetime(df_optimized[col])
        
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        self.logger.info(f"[BRAIN] Оптимandforцandя пам'ятand: {original_memory:.1f}MB  {optimized_memory:.1f}MB "
                        f"(-{memory_reduction:.1f}%)")
        
        return df_optimized
    
    def parallel_collect_data(self, collectors: List[Tuple[str, Any]], 
                           max_workers: int = 3) -> Dict[str, Any]:
        """
        Паралельний withбandр data with рandwithних джерел
        """
        self.logger.info(f"[REFRESH] Паралельний withбandр data with {len(collectors)} джерел...")
        
        results = {}
        
        def collect_data(collector_info):
            name, collector = collector_info
            try:
                start_time = time.time()
                
                # Перевandряємо кеш
                cache_key = f"{name}_data"
                cached_data = self.load_cached_result(cache_key)
                
                if cached_data is not None:
                    self.logger.info(f"[OK] {name}: forванandжено with кешу")
                    return name, cached_data
                
                # Збираємо данand
                if hasattr(collector, 'collect'):
                    data = collector.collect()
                elif callable(collector):
                    data = collector()
                else:
                    raise ValueError(f"Невandдомий тип колектора: {type(collector)}")
                
                # Кешуємо реwithульandт
                self.cache_result(cache_key, data)
                
                elapsed = time.time() - start_time
                self.logger.info(f"[OK] {name}: withandбрано for {elapsed:.1f}s")
                
                return name, data
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error withбandру {name}: {e}")
                return name, None
        
        # [NEW] Паралельnot виконання
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(collect_data, collector_info) 
                      for collector_info in collectors]
            
            for future in futures:
                name, data = future.result()
                if data is not None:
                    results[name] = data
        
        self.logger.info(f"[TARGET] Паралельний withбandр forвершено: {len(results)} джерел")
        
        return results
    
    def incremental_update(self, existing_data: pd.DataFrame, 
                         new_data: pd.DataFrame, 
                         key_columns: List[str] = ['date', 'ticker']) -> pd.DataFrame:
        """
        Інкременandльnot оновлення data
        """
        if existing_data.empty:
            return new_data
        
        if new_data.empty:
            return existing_data
        
        # Знаходимо осandнню дату в andснуючих data
        if 'date' in existing_data.columns:
            last_date = existing_data['date'].max()
            new_data_filtered = new_data[new_data['date'] > last_date]
        else:
            new_data_filtered = new_data
        
        if new_data_filtered.empty:
            self.logger.info(" Нових data for оновлення notмає")
            return existing_data
        
        # Об'єднуємо данand
        updated_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
        
        # Видаляємо дублandкати
        if key_columns:
            updated_data = updated_data.drop_duplicates(subset=key_columns, keep='last')
        
        self.logger.info(f"[UP] Інкременandльnot оновлення: +{len(new_data_filtered)} рядкandв")
        
        return updated_data
    
    def monitor_resources(self) -> Dict[str, float]:
        """
        Монandторинг системних ресурсandв
        """
        return self.resource_monitor.get_current_usage()
    
    def optimize_memory_usage(self):
        """
        Оптимandforцandя викорисandння пам'ятand
        """
        # Очищення notвикорисandної пам'ятand
        gc.collect()
        
        # Логуємо поточnot викорисandння
        usage = self.monitor_resources()
        self.logger.info(f" Викорисandння пам'ятand: {usage['memory_percent']:.1f}%")
        
        if usage['memory_percent'] > 80:
            self.logger.warning("[WARN] Високе викорисandння пам'ятand!")
    
    def cleanup_old_cache(self, max_age_hours: int = 24):
        """
        Очищення сandрого кешу
        """
        current_time = time.time()
        cleaned_files = 0
        
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            file_age = current_time - os.path.getmtime(filepath)
            
            if file_age > max_age_hours * 3600:
                try:
                    os.remove(filepath)
                    cleaned_files += 1
                except Exception as e:
                    self.logger.warning(f"[WARN] Error видалення {filename}: {e}")
        
        if cleaned_files > 0:
            self.logger.info(f" Видалено {cleaned_files} сandрих fileandв кешу")

class ResourceMonitor:
    """
    Монandторинг системних ресурсandв
    """
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger("ResourceMonitor")
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        Поверandє поточnot викорисandння ресурсandв
        """
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Пам'ять
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Диск
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'disk_percent': disk_percent,
                'disk_used_gb': disk_used_gb,
                'disk_total_gb': disk_total_gb
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error монandторингу ресурсandв: {e}")
            return {}
    
    def log_system_status(self):
        """
        Логуємо сandтус system
        """
        usage = self.get_current_usage()
        
        if usage:
            self.logger.info("[DATA] Сandтус system:")
            self.logger.info(f"  - CPU: {usage['cpu_percent']:.1f}%")
            self.logger.info(f"  - Пам'ять: {usage['memory_used_gb']:.1f}/{usage['memory_total_gb']:.1f}GB "
                           f"({usage['memory_percent']:.1f}%)")
            self.logger.info(f"  - Диск: {usage['disk_used_gb']:.1f}/{usage['disk_total_gb']:.1f}GB "
                           f"({usage['disk_percent']:.1f}%)")

# Глобальний екwithемпляр for викорисandння в системand
performance_optimizer = PerformanceOptimizer()

def cache_result(cache_key: str, file_format: str = 'parquet', max_age_hours: Optional[int] = None):
    """
    Декоратор for кешування реwithульandтandв функцandй
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Перевandряємо кеш
            cached_data = performance_optimizer.load_cached_result(cache_key, file_format, max_age_hours)
            
            if cached_data is not None:
                return cached_data
            
            # Виконуємо функцandю
            result = func(*args, **kwargs)
            
            # Кешуємо реwithульandт
            performance_optimizer.cache_result(cache_key, result, file_format)
            
            return result
        
        return wrapper
    return decorator

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Зручна функцandя for оптимandforцandї DataFrame
    """
    return performance_optimizer.optimize_dtypes(df)

def parallel_execute(tasks: List[Tuple[str, Any]], max_workers: int = 3) -> Dict[str, Any]:
    """
    Зручна функцandя for паралельного виконання
    """
    return performance_optimizer.parallel_collect_data(tasks, max_workers)
