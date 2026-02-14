#!/usr/bin/env python3
"""
Memory Optimizer - Оптимізація пам'яті для великих датасетів
"""

import pandas as pd
import numpy as np
import gc
import psutil
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Оптимізація пам'яті для великих data"""
    
    def __init__(self, config: Dict):
        self.memory_limit = config.get('memory', {}).get('limit_gb', 8) * 1024**3
        self.chunk_size = config.get('memory', {}).get('chunk_size', 10000)
        self.auto_gc = config.get('memory', {}).get('auto_gc', True)
        self.monitoring_enabled = config.get('memory', {}).get('monitoring', True)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Отримати поточне використання пам'яті"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимізація DataFrame"""
        if df.empty:
            return df
            
        logger.info(f"Optimizing DataFrame: {df.shape}")
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object to category
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If < 50% unique values
                df[col] = df[col].astype('category')
        
        # Clean up
        df = df.dropna(axis=1, how='all')
        df = df[~df.index.duplicated()]
        
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Optimized to {memory_usage:.2f} MB")
        
        if self.auto_gc:
            gc.collect()
        
        return df
    
    def process_in_chunks(self, df: pd.DataFrame, func, **kwargs):
        """Обробка в чанках"""
        logger.info(f"[RESTART] Processing in chunks: {len(df)} rows, chunk_size: {self.chunk_size}")
        
        results = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size]
            
            # [TARGET] Моніторинг пам'яті
            if self.monitoring_enabled:
                memory_usage = self.get_memory_usage()
                logger.info(f"[DATA] Chunk {i//self.chunk_size + 1}/{total_chunks} - Memory: {memory_usage['rss_mb']:.1f} MB")
                
                # [TARGET] Автоматичне очищення якщо needed
                if memory_usage['rss_mb'] > self.memory_limit / 1024**2 * 0.8:
                    logger.warning("🧠 High memory usage, forcing garbage collection")
                    gc.collect()
            
            try:
                result = func(chunk, **kwargs)
                results.append(result)
                
                # [TARGET] Очищення пам'яті між чанками
                del chunk
                if self.auto_gc:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"[ERROR] Error processing chunk {i//self.chunk_size + 1}: {e}")
                raise
        
        # [TARGET] Об'єднуємо результати
        if results and isinstance(results[0], pd.DataFrame):
            logger.info(f"[RESTART] Concatenating {len(results)} chunks")
            final_result = pd.concat(results, ignore_index=True)
            
            # [TARGET] Фінальна оптимізація
            final_result = self.optimize_dataframe(final_result)
            
            return final_result
        else:
            return results
    
    def check_memory_pressure(self) -> bool:
        """Перевірка тиску на пам'ять"""
        memory_usage = self.get_memory_usage()
        memory_limit_mb = self.memory_limit / 1024**2
        
        return memory_usage['rss_mb'] > memory_limit_mb * 0.8
    
    def force_cleanup(self):
        """Примусове очищення пам'яті"""
        logger.info("🧠 Forcing memory cleanup...")
        gc.collect()
        
        # Очищення pandas кешу
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('mode.chained_assignment', 'warn')
        
        memory_after = self.get_memory_usage()
        logger.info(f"[OK] Memory after cleanup: {memory_after['rss_mb']:.1f} MB")
    
    def suggest_optimizations(self, df: pd.DataFrame) -> List[str]:
        """Підказати оптимізації"""
        suggestions = []
        memory_usage = self.get_memory_usage()
        
        # [TARGET] Аналіз DataFrame
        memory_mb = df.memory_usage(deep=True) / 1024**2
        
        if memory_mb > 1000:
            suggestions.append("Consider processing in chunks")
        
        if len(df) > 100000:
            suggestions.append("Consider downsampling for testing")
        
        # [TARGET] Аналіз типів data
        int_cols = df.select_dtypes(include=['int64']).columns
        if len(int_cols) > 0:
            suggestions.append(f"Downcast {len(int_cols)} integer columns")
        
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            suggestions.append(f"Downcast {len(float_cols)} float columns")
        
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            suggestions.append(f"Convert {len(obj_cols)} object columns to category")
        
        # [TARGET] Аналіз пам'яті
        if memory_usage['rss_mb'] > self.memory_limit / 1024**2 * 0.9:
            suggestions.append("High memory usage detected")
        
        return suggestions
    
    def optimize_memory(self):
        """Оптимізація пам'яті"""
        memory_usage = self.get_memory_usage()
        logger.debug(f"[MEMORY] Current usage: {memory_usage['rss_mb']:.1f} MB")
        
        if self.check_memory_pressure():
            self.force_cleanup()
            return True
        return False


if __name__ == "__main__":
    print("Memory Optimizer - готовий до використання")
    print("🧠 Оптимізація пам'яті для великих датасетів")
