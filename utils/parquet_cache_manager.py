#!/usr/bin/env python3
"""
Parquet Cache Manager - заміна pickle кешу на Parquet
"""

import pandas as pd
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

from utils.parquet_storage import parquet_storage

logger = logging.getLogger(__name__)

class ParquetCacheManager:
    """Менеджер кешу на основі Parquet"""
    
    def __init__(self, cache_dir: str = "cache/parquet"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Типи data для кешування
        self.cache_types = {
            'prices': self.cache_dir / 'prices',
            'news': self.cache_dir / 'news',
            'features': self.cache_dir / 'features',
            'models': self.cache_dir / 'models'
        }
        
        # Створюємо директорії
        for path in self.cache_types.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, data_type: str, tickers: List[str], 
                          timeframes: List[str], **kwargs) -> str:
        """Генерувати ключ кешу"""
        key_data = {
            'type': data_type,
            'tickers': sorted(tickers),
            'timeframes': sorted(timeframes),
            **kwargs
        }
        
        # Створюємо хеш
        key_str = str(sorted(key_data.items()))
        hash_obj = hashlib.md5(key_str.encode())
        return hash_obj.hexdigest()
    
    def get(self, data_type: str, tickers: List[str], timeframes: List[str], 
            max_age_hours: int = 24, **kwargs) -> Optional[pd.DataFrame]:
        """Отримати дані з кешу"""
        try:
            cache_key = self._generate_cache_key(data_type, tickers, timeframes, **kwargs)
            cache_file = self.cache_types[data_type] / f"{cache_key}.parquet"
            
            if not cache_file.exists():
                return None
            
            # Перевіряємо вік кешу
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            if file_age > timedelta(hours=max_age_hours):
                logger.info(f"[ParquetCache] Cache expired for {cache_key}")
                cache_file.unlink()  # Видаляємо застарілий кеш
                return None
            
            # Читаємо дані
            df = pd.read_parquet(cache_file)
            logger.info(f"[ParquetCache] Loaded {len(df)} records from cache")
            return df
            
        except Exception as e:
            logger.error(f"[ParquetCache] Failed to get cache: {e}")
            return None
    
    def set(self, data_type: str, data: pd.DataFrame, tickers: List[str], 
            timeframes: List[str], **kwargs) -> str:
        """Зберегти дані в кеш"""
        try:
            cache_key = self._generate_cache_key(data_type, tickers, timeframes, **kwargs)
            cache_file = self.cache_types[data_type] / f"{cache_key}.parquet"
            
            # Додаємо метадані
            data = data.copy()
            data['cached_at'] = datetime.now(timezone.utc)
            data['cache_key'] = cache_key
            
            # Зберігаємо в Parquet
            data.to_parquet(cache_file, compression='snappy', index=False)
            
            logger.info(f"[ParquetCache] Cached {len(data)} records to {cache_key}")
            return str(cache_file)
            
        except Exception as e:
            logger.error(f"[ParquetCache] Failed to set cache: {e}")
            raise
    
    def invalidate(self, data_type: str = None, pattern: str = None):
        """Інвалідувати кеш"""
        try:
            if data_type:
                # Видаляємо весь тип data
                cache_path = self.cache_types[data_type]
                for file_path in cache_path.glob("*.parquet"):
                    file_path.unlink()
                logger.info(f"[ParquetCache] Invalidated {data_type} cache")
            
            elif pattern:
                # Видаляємо за шаблоном
                for cache_type, cache_path in self.cache_types.items():
                    for file_path in cache_path.glob(f"*{pattern}*.parquet"):
                        file_path.unlink()
                logger.info(f"[ParquetCache] Invalidated cache matching: {pattern}")
            
            else:
                # Видаляємо весь кеш
                for cache_path in self.cache_types.values():
                    for file_path in cache_path.glob("*.parquet"):
                        file_path.unlink()
                logger.info("[ParquetCache] Invalidated all cache")
                
        except Exception as e:
            logger.error(f"[ParquetCache] Failed to invalidate cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Отримати статистику кешу"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'by_type': {}
        }
        
        for cache_type, cache_path in self.cache_types.items():
            files = list(cache_path.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in files)
            
            stats['by_type'][cache_type] = {
                'file_count': len(files),
                'size_mb': total_size / (1024 * 1024)
            }
            
            stats['total_files'] += len(files)
            stats['total_size_mb'] += total_size / (1024 * 1024)
        
        return stats
    
    def cleanup_expired(self, max_age_hours: int = 24):
        """Очистити застарілий кеш"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            for cache_path in self.cache_types.values():
                for file_path in cache_path.glob("*.parquet"):
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
            
            logger.info(f"[ParquetCache] Cleaned {cleaned_count} expired cache files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"[ParquetCache] Failed to cleanup: {e}")
            return 0
    
    def migrate_from_pickle(self, pickle_cache_dir: str = "cache"):
        """Мігрувати дані з pickle кешу"""
        try:
            pickle_dir = Path(pickle_cache_dir)
            if not pickle_dir.exists():
                logger.info(f"[ParquetCache] Pickle cache dir not found: {pickle_dir}")
                return
            
            pkl_files = list(pickle_dir.glob("*.pkl"))
            logger.info(f"[ParquetCache] Found {len(pkl_files)} pickle files to migrate")
            
            migrated_count = 0
            
            for pkl_file in pkl_files:
                try:
                    # Пропускаємо малі файли
                    if pkl_file.stat().st_size < 1000:
                        continue
                    
                    # Читаємо pickle
                    import pickle
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, pd.DataFrame):
                        # Визначаємо тип data
                        if 'ticker' in data.columns or 'Open' in data.columns:
                            data_type = 'prices'
                            tickers = data.get('ticker', ['UNKNOWN']).unique().tolist()
                            timeframes = ['1d']
                        elif 'title' in data.columns:
                            data_type = 'news'
                            tickers = ['all']
                            timeframes = ['all']
                        else:
                            data_type = 'features'
                            tickers = ['all']
                            timeframes = ['all']
                        
                        # Зберігаємо в Parquet кеш
                        self.set(data_type, data, tickers, timeframes)
                        migrated_count += 1
                        
                        # Видаляємо оригінальний pickle
                        pkl_file.unlink()
                    
                except Exception as e:
                    logger.warning(f"[ParquetCache] Failed to migrate {pkl_file.name}: {e}")
            
            logger.info(f"[ParquetCache] Migrated {migrated_count} files from pickle")
            return migrated_count
            
        except Exception as e:
            logger.error(f"[ParquetCache] Migration failed: {e}")
            return 0

# Глобальний інстанс
parquet_cache = ParquetCacheManager()
