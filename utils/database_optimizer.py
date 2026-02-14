#!/usr/bin/env python3
"""
Database Optimizer - Оптимізація бази data
Відповідно до плану з database_optimization_plan.md
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import shutil
import gzip
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """Оптимізатор бази data"""
    
    def __init__(self):
        self.data_path = Path("c:/trading_project/data")
        self.db_path = self.data_path / "databases"
        self.cache_path = self.data_path / "cache"
        self.processed_path = self.data_path / "processed"
        self.archive_path = self.data_path / "archive"
        self.metadata_path = self.data_path / "metadata"
        
        # Створюємо структуру
        self._create_optimized_structure()
        
    def _create_optimized_structure(self):
        """Створення оптимізованої структури директорій"""
        directories = [
            self.metadata_path,
            self.archive_path / "daily",
            self.archive_path / "weekly", 
            self.archive_path / "monthly",
            self.data_path / "raw" / "news",
            self.data_path / "raw" / "prices",
            self.data_path / "raw" / "macro",
            self.data_path / "processed" / "stage1",
            self.data_path / "processed" / "stage2",
            self.data_path / "processed" / "stage3",
            self.data_path / "processed" / "stage4",
            self.data_path / "models" / "production",
            self.data_path / "models" / "staging",
            self.data_path / "models" / "archive",
            self.data_path / "cache" / "queries",
            self.data_path / "cache" / "computations",
            self.data_path / "cache" / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def analyze_current_database(self) -> Dict[str, Any]:
        """Аналіз поточного стану бази data"""
        logger.info("Analyzing current database structure...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_size': 0,
            'file_count': 0,
            'database_files': {},
            'cache_files': {},
            'processed_files': {},
            'issues': [],
            'recommendations': []
        }
        
        # Аналіз баз data
        for db_file in self.db_path.glob("*.db"):
            size = db_file.stat().st_size
            analysis['database_files'][db_file.name] = {
                'size': size,
                'size_mb': round(size / (1024*1024), 2),
                'modified': datetime.fromtimestamp(db_file.stat().st_mtime).isoformat()
            }
            analysis['total_size'] += size
            analysis['file_count'] += 1
            
        # Аналіз кешу
        cache_files = list(self.cache_path.rglob("*"))
        analysis['cache_files'] = {
            'count': len(cache_files),
            'size_mb': round(sum(f.stat().st_size for f in cache_files if f.is_file()) / (1024*1024), 2)
        }
        
        # Аналіз оброблених файлів
        processed_files = list(self.processed_path.rglob("*.parquet"))
        analysis['processed_files'] = {
            'count': len(processed_files),
            'size_mb': round(sum(f.stat().st_size for f in processed_files) / (1024*1024), 2)
        }
        
        # Виявлення проблем
        if analysis['cache_files']['count'] > 5000:
            analysis['issues'].append(f"Too many cache files: {analysis['cache_files']['count']}")
            analysis['recommendations'].append("Clean up old cache files")
            
        if analysis['total_size'] > 5 * 1024**3:  # 5GB
            analysis['issues'].append(f"Database size too large: {analysis['total_size']/(1024**3):.2f}GB")
            analysis['recommendations'].append("Archive old data")
            
        return analysis
        
    def optimize_sqlite_databases(self) -> Dict[str, Any]:
        """Оптимізація SQLite баз data"""
        logger.info("Optimizing SQLite databases...")
        
        results = {
            'optimized': [],
            'errors': [],
            'total_space_saved': 0
        }
        
        for db_file in self.db_path.glob("*.db"):
            try:
                original_size = db_file.stat().st_size
                
                # VACUUM для оптимізації
                conn = sqlite3.connect(str(db_file))
                conn.execute("VACUUM")
                
                # ANALYZE для оновлення статистики
                conn.execute("ANALYZE")
                
                # Перевірка індексів
                cursor = conn.cursor()
                cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index'")
                indexes = cursor.fetchall()
                
                conn.close()
                
                new_size = db_file.stat().st_size
                space_saved = original_size - new_size
                
                results['optimized'].append({
                    'database': db_file.name,
                    'original_size_mb': round(original_size / (1024*1024), 2),
                    'new_size_mb': round(new_size / (1024*1024), 2),
                    'space_saved_mb': round(space_saved / (1024*1024), 2),
                    'indexes_count': len(indexes)
                })
                
                results['total_space_saved'] += space_saved
                
            except Exception as e:
                results['errors'].append({
                    'database': db_file.name,
                    'error': str(e)
                })
                
        return results
        
    def compress_parquet_files(self) -> Dict[str, Any]:
        """Стиснення Parquet файлів"""
        logger.info("Compressing Parquet files...")
        
        results = {
            'compressed': [],
            'errors': [],
            'total_space_saved': 0
        }
        
        for parquet_file in self.processed_path.rglob("*.parquet"):
            try:
                original_size = parquet_file.stat().st_size
                
                # Читання оригінального файлу
                df = pd.read_parquet(parquet_file)
                
                # Стиснення з Snappy
                compressed_path = parquet_file.with_suffix('.compressed.parquet')
                df.to_parquet(compressed_path, compression='snappy', index=False)
                
                compressed_size = compressed_path.stat().st_size
                space_saved = original_size - compressed_size
                
                # Якщо стиснення ефективне, замінюємо файл
                if space_saved > 0:
                    backup_path = parquet_file.with_suffix('.backup.parquet')
                    shutil.move(str(parquet_file), str(backup_path))
                    shutil.move(str(compressed_path), str(parquet_file))
                    backup_path.unlink()  # Видаляємо бекап
                    
                    results['compressed'].append({
                        'file': str(parquet_file.relative_to(self.data_path)),
                        'original_size_mb': round(original_size / (1024*1024), 2),
                        'compressed_size_mb': round(compressed_size / (1024*1024), 2),
                        'space_saved_mb': round(space_saved / (1024*1024), 2),
                        'compression_ratio': round(compressed_size / original_size, 3)
                    })
                    
                    results['total_space_saved'] += space_saved
                else:
                    compressed_path.unlink()  # Видаляємо неефективне стиснення
                    
            except Exception as e:
                results['errors'].append({
                    'file': str(parquet_file.relative_to(self.data_path)),
                    'error': str(e)
                })
                
        return results
        
    def cleanup_cache_files(self, days_old: int = 7) -> Dict[str, Any]:
        """Очищення старих кеш-файлів"""
        logger.info(f"Cleaning up cache files older than {days_old} days...")
        
        results = {
            'deleted': [],
            'errors': [],
            'total_space_freed': 0,
            'files_deleted': 0
        }
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for cache_file in self.cache_path.rglob("*"):
            if cache_file.is_file():
                try:
                    if cache_file.stat().st_mtime < cutoff_time:
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        
                        results['deleted'].append({
                            'file': str(cache_file.relative_to(self.data_path)),
                            'size_mb': round(file_size / (1024*1024), 2)
                        })
                        
                        results['total_space_freed'] += file_size
                        results['files_deleted'] += 1
                        
                except Exception as e:
                    results['errors'].append({
                        'file': str(cache_file.relative_to(self.data_path)),
                        'error': str(e)
                    })
                    
        return results
        
    def archive_old_data(self, days_old: int = 30) -> Dict[str, Any]:
        """Архівування старих data"""
        logger.info(f"Archiving data older than {days_old} days...")
        
        results = {
            'archived': [],
            'errors': [],
            'total_space_freed': 0
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archive_dir = self.archive_path / "daily" / cutoff_date.strftime("%Y/%m/%d")
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Архівування старих Parquet файлів
        for parquet_file in self.processed_path.rglob("*.parquet"):
            try:
                file_mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    archive_path = archive_dir / parquet_file.name
                    
                    # Стиснення перед архівуванням
                    with open(parquet_file, 'rb') as f_in:
                        with gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    original_size = parquet_file.stat().st_size
                    compressed_size = (archive_path.parent / f"{parquet_file.name}.gz").stat().st_size
                    
                    results['archived'].append({
                        'file': str(parquet_file.relative_to(self.data_path)),
                        'archive_path': str(archive_path.relative_to(self.data_path)),
                        'original_size_mb': round(original_size / (1024*1024), 2),
                        'compressed_size_mb': round(compressed_size / (1024*1024), 2)
                    })
                    
                    results['total_space_freed'] += original_size
                    parquet_file.unlink()  # Видаляємо оригінал
                    
            except Exception as e:
                results['errors'].append({
                    'file': str(parquet_file.relative_to(self.data_path)),
                    'error': str(e)
                })
                
        return results
        
    def create_metadata_schema(self) -> Dict[str, Any]:
        """Створення схеми метаdata"""
        logger.info("Creating metadata schema...")
        
        metadata = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'database_structure': {
                'raw': {
                    'description': 'Raw data from collectors',
                    'subdirectories': ['news', 'prices', 'macro', 'insider']
                },
                'processed': {
                    'description': 'Processed data by stages',
                    'subdirectories': ['stage1', 'stage2', 'stage3', 'stage4']
                },
                'models': {
                    'description': 'Trained models',
                    'subdirectories': ['production', 'staging', 'archive']
                },
                'cache': {
                    'description': 'Temporary cache files',
                    'subdirectories': ['queries', 'computations', 'temp']
                },
                'archive': {
                    'description': 'Archived data',
                    'subdirectories': ['daily', 'weekly', 'monthly']
                }
            },
            'data_schemas': {
                'stage1': {
                    'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    'column_types': {
                        'timestamp': 'datetime64[ns]',
                        'open': 'float64',
                        'high': 'float64',
                        'low': 'float64',
                        'close': 'float64',
                        'volume': 'int64'
                    }
                },
                'stage2': {
                    'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'features'],
                    'column_types': {
                        'timestamp': 'datetime64[ns]',
                        'open': 'float64',
                        'high': 'float64',
                        'low': 'float64',
                        'close': 'float64',
                        'volume': 'int64',
                        'features': 'object'
                    }
                }
            }
        }
        
        # Збереження метаdata
        schema_file = self.metadata_path / "schema.json"
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return {'schema_file': str(schema_file), 'metadata': metadata}
        
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Генерація звіту оптимізації"""
        logger.info("Generating optimization report...")
        
        # Аналіз поточного стану
        analysis = self.analyze_current_database()
        
        # Оптимізація баз data
        db_optimization = self.optimize_sqlite_databases()
        
        # Стиснення файлів
        compression_results = self.compress_parquet_files()
        
        # Очищення кешу
        cache_cleanup = self.cleanup_cache_files()
        
        # Архівування
        archive_results = self.archive_old_data()
        
        # Створення метаdata
        metadata_schema = self.create_metadata_schema()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'optimization_results': {
                'database_optimization': db_optimization,
                'compression': compression_results,
                'cache_cleanup': cache_cleanup,
                'archiving': archive_results,
                'metadata_schema': metadata_schema
            },
            'summary': {
                'total_space_saved': (
                    db_optimization.get('total_space_saved', 0) +
                    compression_results.get('total_space_saved', 0) +
                    cache_cleanup.get('total_space_freed', 0) +
                    archive_results.get('total_space_freed', 0)
                ) / (1024*1024),  # MB
                'files_processed': (
                    len(db_optimization.get('optimized', [])) +
                    len(compression_results.get('compressed', [])) +
                    cache_cleanup.get('files_deleted', 0) +
                    len(archive_results.get('archived', []))
                ),
                'errors_count': (
                    len(db_optimization.get('errors', [])) +
                    len(compression_results.get('errors', [])) +
                    len(cache_cleanup.get('errors', [])) +
                    len(archive_results.get('errors', []))
                )
            }
        }
        
        # Збереження звіту
        report_file = self.metadata_path / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Optimization report saved to {report_file}")
        return report
        
    def run_full_optimization(self) -> Dict[str, Any]:
        """Запуск повної оптимізації"""
        logger.info("Starting full database optimization...")
        
        start_time = time.time()
        
        try:
            report = self.generate_optimization_report()
            
            end_time = time.time()
            duration = end_time - start_time
            
            summary = {
                'status': 'completed',
                'duration_seconds': round(duration, 2),
                'space_saved_mb': round(report['summary']['total_space_saved'], 2),
                'files_processed': report['summary']['files_processed'],
                'errors_count': report['summary']['errors_count']
            }
            
            logger.info(f"Optimization completed in {duration:.2f}s")
            logger.info(f"Space saved: {summary['space_saved_mb']} MB")
            logger.info(f"Files processed: {summary['files_processed']}")
            
            return {
                'summary': summary,
                'detailed_report': report
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'summary': {
                    'status': 'failed',
                    'error': str(e),
                    'duration_seconds': round(time.time() - start_time, 2)
                }
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    optimizer = DatabaseOptimizer()
    result = optimizer.run_full_optimization()
    
    print("\n" + "="*50)
    print("DATABASE OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Status: {result['summary']['status']}")
    print(f"Duration: {result['summary'].get('duration_seconds', 0)} seconds")
    
    if result['summary']['status'] == 'completed':
        print(f"Space Saved: {result['summary']['space_saved_mb']} MB")
        print(f"Files Processed: {result['summary']['files_processed']}")
        print(f"Errors: {result['summary']['errors_count']}")
    else:
        print(f"Error: {result['summary'].get('error', 'Unknown error')}")
    
    print("="*50)
