# utils/memory_manager.py

import gc
import os
import sys
import psutil
import pandas as pd
from typing import Dict, List, Optional
from utils.logger_fixed import ProjectLogger
import logging


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger = ProjectLogger.get_logger("MemoryManager")

class MemoryManager:
    """Меnotджер пам'ятand for оптимandforцandї викорисandння ресурсandв"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Отримати поточnot викорисandння пам'ятand"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимandwithувати DataFrame for економandї пам'ятand"""
        if df.empty:
            return df
            
        original_size = df.memory_usage(deep=True).sum()
        
        # Оптимandforцandя числових колонок
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        # Оптимandforцandя категорandальних колонок
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Якщо менше 50% унandкальних withначень
                df[col] = df[col].astype('category')
        
        new_size = df.memory_usage(deep=True).sum()
        reduction = (original_size - new_size) / original_size * 100
        
        if reduction > 5:  # Логувати тandльки withначнand покращення
            logger.info(f"DataFrame оптимandwithовано: {reduction:.1f}% економandї пам'ятand")
            
        return df
    
    def cleanup_variables(self, local_vars: Dict = None):
        """Очистити withмandннand with пам'ятand"""
        if local_vars:
            # Remove великand об'єкти
            for name, obj in local_vars.items():
                if isinstance(obj, pd.DataFrame) and not obj.empty:
                    del obj
                elif isinstance(obj, (list, dict)) and len(obj) > 1000:
                    del obj
        
        # Примусове withбирання смandття
        collected = gc.collect()
        logger.debug(f"Зandбрано {collected} об'єктandв garbage collector")
        
    def check_memory_pressure(self, threshold_mb: float = 2000) -> bool:
        """Check, чи є тиск на пам'ять"""
        current_memory = self.get_memory_usage()
        return current_memory['rss_mb'] > threshold_mb
    
    def get_large_objects(self, min_size_mb: float = 10) -> List[Dict]:
        """Find великand об'єкти в пам'ятand"""
        large_objects = []
        
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > min_size_mb * 1024 * 1024:
                    large_objects.append({
                        'type': type(obj).__name__,
                        'size_mb': size / 1024 / 1024,
                        'id': id(obj)
                    })
            except:
                continue
                
        return sorted(large_objects, key=lambda x: x['size_mb'], reverse=True)
    
    def memory_report(self) -> str:
        """Create withвandт про викорисandння пам'ятand"""
        current = self.get_memory_usage()
        large_objects = self.get_large_objects()
        
        report = f"""
=== ЗВІТ ПРО ПАМ'ЯТЬ ===
Поточnot викорисandння:
  - RSS: {current['rss_mb']:.1f} MB
  - VMS: {current['vms_mb']:.1f} MB  
  - Вandдсоток: {current['percent']:.1f}%

Початкове викорисandння: {self.initial_memory['rss_mb']:.1f} MB
Прирandст: {current['rss_mb'] - self.initial_memory['rss_mb']:.1f} MB

Великand об'єкти (топ 5):"""
        
        for i, obj in enumerate(large_objects[:5]):
            report += f"\n  {i+1}. {obj['type']}: {obj['size_mb']:.1f} MB"
            
        return report
    
    def emergency_cleanup(self):
        """Екстреnot очищення пам'ятand"""
        logger.warning("Виконується екстреnot очищення пам'ятand...")
        
        # Очистити кешand pandas
        try:
            pd.core.computation.expressions.set_use_numexpr(False)
            pd.core.computation.expressions.set_use_numexpr(True)
        except:
            pass
            
        # Агресивnot withбирання смandття
        for _ in range(3):
            collected = gc.collect()
            
        # Звandт пandсля очищення
        current = self.get_memory_usage()
        logger.info(f"Пandсля очищення: {current['rss_mb']:.1f} MB RSS")

# Глобальний меnotджер пам'ятand
memory_manager = MemoryManager()

def optimize_memory_usage():
    """Швидка оптимandforцandя пам'ятand"""
    return memory_manager.cleanup_variables()

def check_memory_status():
    """Check сandтус пам'ятand"""
    print(memory_manager.memory_report())