#!/usr/bin/env python3
"""
Parquet Storage Manager - оптимізоване зберігання data
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

class ParquetStorageManager:
    """Менеджер зберігання data у форматі Parquet"""
    
    def __init__(self, base_path: str = "data/parquet"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Структура директорій
        self.paths = {
            'prices': self.base_path / 'prices',
            'news': self.base_path / 'news', 
            'sentiment': self.base_path / 'sentiment',
            'features': self.base_path / 'features',
            'models': self.base_path / 'models'
        }
        
        # Створюємо директорії
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def save_prices(self, df: pd.DataFrame, ticker: str, timeframe: str, 
                   partition_cols: List[str] = None) -> str:
        """Зберегти цінові дані з партиціями"""
        try:
            # Перевіряємо та очищуємо дані
            df = self._clean_dataframe(df)
            
            # Додаємо метадані
            df['ticker'] = ticker
            df['timeframe'] = timeframe
            df['saved_at'] = datetime.now(timezone.utc)
            
            # Шлях для збереження
            file_path = self.paths['prices'] / f"{ticker}_{timeframe}.parquet"
            
            # Партиції для оптимізації
            if partition_cols is None:
                partition_cols = ['ticker', 'timeframe']
            
            # Зберігаємо з компресією
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                file_path,
                compression='snappy',
                write_statistics=True
            )
            
            logger.info(f"[Parquet] Saved {len(df)} price records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"[Parquet] Failed to save prices: {e}")
            raise
    
    def load_prices(self, ticker: str = None, timeframe: str = None, 
                   start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Завантажити цінові дані з фільтрами"""
        try:
            # Формуємо шлях
            if ticker and timeframe:
                file_path = self.paths['prices'] / f"{ticker}_{timeframe}.parquet"
                if not file_path.exists():
                    logger.warning(f"[Parquet] File not found: {file_path}")
                    return pd.DataFrame()
                
                # Читаємо дані
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                # Фільтруємо по датах
                if start_date:
                    df = df[df['Datetime'] >= start_date]
                if end_date:
                    df = df[df['Datetime'] <= end_date]
                
                logger.info(f"[Parquet] Loaded {len(df)} price records")
                return df
            
            # Якщо не вказано тікер/таймфрейм - читаємо всі файли
            all_data = []
            for file_path in self.paths['prices'].glob("*.parquet"):
                try:
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    
                    # Фільтруємо
                    if ticker and ticker not in file_path.stem:
                        continue
                    if timeframe and timeframe not in file_path.stem:
                        continue
                    
                    all_data.append(df)
                except Exception as e:
                    logger.warning(f"[Parquet] Failed to read {file_path}: {e}")
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                logger.info(f"[Parquet] Loaded {len(result)} total price records")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[Parquet] Failed to load prices: {e}")
            return pd.DataFrame()
    
    def save_news(self, df: pd.DataFrame, source: str = None) -> str:
        """Зберегти новини"""
        try:
            df = self._clean_dataframe(df)
            
            if source:
                df['source'] = source
            
            df['saved_at'] = datetime.now(timezone.utc)
            
            file_path = self.paths['news'] / f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                file_path,
                compression='snappy',
                write_statistics=True
            )
            
            logger.info(f"[Parquet] Saved {len(df)} news records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"[Parquet] Failed to save news: {e}")
            raise
    
    def load_news(self, source: str = None, start_date: str = None, 
                 limit: int = None) -> pd.DataFrame:
        """Завантажити новини"""
        try:
            all_data = []
            
            for file_path in self.paths['news'].glob("news_*.parquet"):
                try:
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    
                    # Фільтруємо
                    if source and 'source' in df.columns:
                        df = df[df['source'] == source]
                    
                    all_data.append(df)
                except Exception as e:
                    logger.warning(f"[Parquet] Failed to read {file_path}: {e}")
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                
                # Сортуємо по даті
                if 'published_at' in result.columns:
                    result = result.sort_values('published_at', ascending=False)
                
                # Ліміт
                if limit:
                    result = result.head(limit)
                
                logger.info(f"[Parquet] Loaded {len(result)} news records")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[Parquet] Failed to load news: {e}")
            return pd.DataFrame()
    
    def save_sentiment(self, df: pd.DataFrame, analyzer: str) -> str:
        """Зберегти сентимент аналіз"""
        try:
            df = self._clean_dataframe(df)
            df['analyzer'] = analyzer
            df['saved_at'] = datetime.now(timezone.utc)
            
            file_path = self.paths['sentiment'] / f"sentiment_{analyzer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                file_path,
                compression='snappy',
                write_statistics=True
            )
            
            logger.info(f"[Parquet] Saved {len(df)} sentiment records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"[Parquet] Failed to save sentiment: {e}")
            raise
    
    def load_sentiment(self, analyzer: str = None, limit: int = None) -> pd.DataFrame:
        """Завантажити сентимент аналіз"""
        try:
            all_data = []
            
            for file_path in self.paths['sentiment'].glob("sentiment_*.parquet"):
                try:
                    if analyzer and analyzer not in file_path.stem:
                        continue
                    
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    all_data.append(df)
                except Exception as e:
                    logger.warning(f"[Parquet] Failed to read {file_path}: {e}")
            
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                
                if limit:
                    result = result.head(limit)
                
                logger.info(f"[Parquet] Loaded {len(result)} sentiment records")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[Parquet] Failed to load sentiment: {e}")
            return pd.DataFrame()
    
    def save_features(self, df: pd.DataFrame, ticker: str, timeframe: str) -> str:
        """Зберегти фічі"""
        try:
            df = self._clean_dataframe(df)
            df['ticker'] = ticker
            df['timeframe'] = timeframe
            df['saved_at'] = datetime.now(timezone.utc)
            
            file_path = self.paths['features'] / f"features_{ticker}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                file_path,
                compression='snappy',
                write_statistics=True
            )
            
            logger.info(f"[Parquet] Saved {len(df)} feature records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"[Parquet] Failed to save features: {e}")
            raise
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Отримати статистику зберігання"""
        stats = {
            'total_size_mb': 0,
            'file_counts': {},
            'data_types': {}
        }
        
        for data_type, path in self.paths.items():
            files = list(path.glob("**/*.parquet"))
            stats['file_counts'][data_type] = len(files)
            
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            stats['data_types'][data_type] = {
                'size_mb': total_size / (1024 * 1024),
                'file_count': len(files)
            }
            stats['total_size_mb'] += total_size / (1024 * 1024)
        
        return stats
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистити DataFrame для збереження"""
        # Видаляємо NaN в критичних колонках
        if 'Datetime' in df.columns:
            df = df.dropna(subset=['Datetime'])
        
        # Конвертуємо типи
        for col in df.columns:
            if df[col].dtype == 'object':
                # Спробуємо конвертувати в кращий тип
                try:
                    if col == 'Datetime':
                        df[col] = pd.to_datetime(df[col])
                    elif col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def migrate_from_csv(self, csv_path: str, data_type: str, 
                        ticker: str = None, timeframe: str = None) -> str:
        """Мігрувати дані з CSV в Parquet"""
        try:
            logger.info(f"[Parquet] Migrating {csv_path} to Parquet...")
            
            # Читаємо CSV
            df = pd.read_csv(csv_path)
            
            # Зберігаємо в Parquet
            if data_type == 'prices':
                return self.save_prices(df, ticker, timeframe)
            elif data_type == 'news':
                return self.save_news(df)
            elif data_type == 'features':
                return self.save_features(df, ticker, timeframe)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
                
        except Exception as e:
            logger.error(f"[Parquet] Migration failed: {e}")
            raise
    
    def cleanup_old_files(self, days_old: int = 30):
        """Очистити старі файли"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
            
            for path in self.paths.values():
                for file_path in path.glob("**/*.parquet"):
                    if file_path.stat().st_mtime < cutoff_date:
                        file_path.unlink()
                        logger.info(f"[Parquet] Deleted old file: {file_path}")
                        
        except Exception as e:
            logger.error(f"[Parquet] Cleanup failed: {e}")

# Глобальний інстанс
parquet_storage = ParquetStorageManager()
