#!/usr/bin/env python3
"""
Batch Processing Manager для великих наборів тікерів
Оптимізований для швидкої обробки 50+ тікерів
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config.tickers import get_tickers, get_category_stats, validate_tickers


@dataclass
class BatchConfig:
    """Конфігурація пакетної обробки"""
    # Розміри батчів
    small_batch_size: int = 5      # Маленький батч
    medium_batch_size: int = 10    # Середній батч
    large_batch_size: int = 20     # Великий батч
    max_batch_size: int = 50       # Максимальний батч
    
    # Ресурси
    max_workers: int = 4            # Кількість потоків
    memory_limit_gb: float = 8.0    # Ліміт пам'яті
    timeout_seconds: int = 300      # Таймаут на батч
    
    # Оптимізація
    enable_parallel: bool = True    # Паралельна обробка
    enable_caching: bool = True     # Кешування
    enable_monitoring: bool = True  # Моніторинг прогресу
    
    # Стратегії
    strategy: str = "adaptive"      # adaptive, fixed, priority
    priority_categories: List[str] = field(default_factory=lambda: ["tech", "finance", "core"])


class BatchProcessor:
    """Менеджер пакетної обробки тікерів"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.logger = logging.getLogger("BatchProcessor")
        
        # Створюємо директорії
        self.cache_dir = Path("cache/batch")
        self.results_dir = Path("results/batch")
        self.monitoring_dir = Path("analytics/batch")
        
        for dir_path in [self.cache_dir, self.results_dir, self.monitoring_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Стан обробки
        self.processing_stats = {
            'total_tickers': 0,
            'processed_tickers': 0,
            'failed_tickers': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None,
            'processing_time': 0.0
        }
        
        # Блокування для потокоwithoutпеки
        self._lock = threading.Lock()
        
        self.logger.info("BatchProcessor initialized")
    
    def create_optimal_batches(self, tickers: List[str]) -> List[List[str]]:
        """
        Створити оптимальні батчі для обробки
        
        Args:
            tickers: Список тікерів
            
        Returns:
            List[List[str]]: Список батчів
        """
        if not tickers:
            return []
        
        self.logger.info(f"Creating optimal batches for {len(tickers)} tickers")
        
        if self.config.strategy == "adaptive":
            return self._create_adaptive_batches(tickers)
        elif self.config.strategy == "priority":
            return self._create_priority_batches(tickers)
        else:
            return self._create_fixed_batches(tickers)
    
    def _create_adaptive_batches(self, tickers: List[str]) -> List[List[str]]:
        """Адаптивне створення батчів"""
        total_tickers = len(tickers)
        
        # Визначаємо оптимальний розмір батчу
        if total_tickers <= 10:
            batch_size = self.config.small_batch_size
        elif total_tickers <= 30:
            batch_size = self.config.medium_batch_size
        elif total_tickers <= 60:
            batch_size = self.config.large_batch_size
        else:
            batch_size = self.config.max_batch_size
        
        # Створюємо батчі
        batches = []
        for i in range(0, total_tickers, batch_size):
            batch = tickers[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} adaptive batches (size: {batch_size})")
        return batches
    
    def _create_priority_batches(self, tickers: List[str]) -> List[List[str]]:
        """Створення батчів за пріоритетом"""
        from config.tickers import get_ticker_categories
        
        # Сортуємо тікери за пріоритетом категорій
        priority_tickers = []
        other_tickers = []
        
        for ticker in tickers:
            categories = get_ticker_categories(ticker)
            if any(cat in self.config.priority_categories for cat in categories):
                priority_tickers.append(ticker)
            else:
                other_tickers.append(ticker)
        
        # Створюємо батчі: пріоритетні перші
        all_tickers = priority_tickers + other_tickers
        return self._create_fixed_batches(all_tickers)
    
    def _create_fixed_batches(self, tickers: List[str]) -> List[List[str]]:
        """Фіксоване створення батчів"""
        batch_size = self.config.medium_batch_size
        batches = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} fixed batches (size: {batch_size})")
        return batches
    
    def process_batches(self, batches: List[List[str]], 
                       processing_func: callable) -> Dict[str, Any]:
        """
        Обробити батчі
        
        Args:
            batches: Список батчів
            processing_func: Функція обробки
            
        Returns:
            Dict[str, Any]: Результати обробки
        """
        if not batches:
            return {'status': 'error', 'message': 'No batches to process'}
        
        self.processing_stats['total_tickers'] = sum(len(batch) for batch in batches)
        self.processing_stats['start_time'] = time.time()
        
        self.logger.info(f"Starting batch processing: {len(batches)} batches, "
                        f"{self.processing_stats['total_tickers']} total tickers")
        
        results = []
        
        if self.config.enable_parallel and len(batches) > 1:
            results = self._process_batches_parallel(batches, processing_func)
        else:
            results = self._process_batches_sequential(batches, processing_func)
        
        self.processing_stats['end_time'] = time.time()
        self.processing_stats['processing_time'] = (
            self.processing_stats['end_time'] - self.processing_stats['start_time']
        )
        
        # Формуємо фінальні результати
        final_results = {
            'status': 'success',
            'batches_processed': len(results),
            'total_tickers': self.processing_stats['total_tickers'],
            'processed_tickers': self.processing_stats['processed_tickers'],
            'failed_tickers': self.processing_stats['failed_tickers'],
            'processing_time': self.processing_stats['processing_time'],
            'batch_results': results,
            'success_rate': (self.processing_stats['processed_tickers'] / 
                           self.processing_stats['total_tickers']) if self.processing_stats['total_tickers'] > 0 else 0
        }
        
        self.logger.info(f"Batch processing completed: {final_results['success_rate']:.2%} success rate")
        
        # Зберігаємо результати
        self._save_results(final_results)
        
        return final_results
    
    def _process_batches_parallel(self, batches: List[List[str]], 
                                 processing_func: callable) -> List[Dict[str, Any]]:
        """Паралельна обробка батчів"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Створюємо future'и
            future_to_batch = {
                executor.submit(self._process_single_batch, batch, processing_func): batch 
                for batch in batches
            }
            
            # Обробляємо результати
            for future in as_completed(future_to_batch, timeout=self.config.timeout_seconds):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Оновлюємо статистику
                    with self._lock:
                        self.processing_stats['processed_tickers'] += result.get('processed_count', 0)
                        self.processing_stats['failed_tickers'] += result.get('failed_count', 0)
                        self.processing_stats['batches_processed'] += 1
                    
                    # Виводимо прогрес
                    progress = (self.processing_stats['batches_processed'] / len(batches)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% - Batch {self.processing_stats['batches_processed']}/{len(batches)}")
                    
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    results.append({
                        'batch': batch,
                        'status': 'failed',
                        'error': str(e),
                        'processed_count': 0,
                        'failed_count': len(batch)
                    })
        
        return results
    
    def _process_batches_sequential(self, batches: List[List[str]], 
                                  processing_func: callable) -> List[Dict[str, Any]]:
        """Послідовна обробка батчів"""
        results = []
        
        for i, batch in enumerate(batches):
            try:
                result = self._process_single_batch(batch, processing_func)
                results.append(result)
                
                # Оновлюємо статистику
                self.processing_stats['processed_tickers'] += result.get('processed_count', 0)
                self.processing_stats['failed_tickers'] += result.get('failed_count', 0)
                self.processing_stats['batches_processed'] += 1
                
                # Виводимо прогрес
                progress = ((i + 1) / len(batches)) * 100
                self.logger.info(f"Progress: {progress:.1f}% - Batch {i + 1}/{len(batches)}")
                
            except Exception as e:
                self.logger.error(f"Batch {i + 1} processing failed: {e}")
                results.append({
                    'batch': batch,
                    'status': 'failed',
                    'error': str(e),
                    'processed_count': 0,
                    'failed_count': len(batch)
                })
        
        return results
    
    def _process_single_batch(self, batch: List[str], processing_func: callable) -> Dict[str, Any]:
        """Обробка одного батчу"""
        start_time = time.time()
        
        try:
            # Обробляємо батч
            batch_results = processing_func(batch)
            
            processing_time = time.time() - start_time
            
            return {
                'batch': batch,
                'status': 'success',
                'results': batch_results,
                'processed_count': len(batch),
                'failed_count': 0,
                'processing_time': processing_time,
                'tickers_per_second': len(batch) / processing_time if processing_time > 0 else 0
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return {
                'batch': batch,
                'status': 'failed',
                'error': str(e),
                'processed_count': 0,
                'failed_count': len(batch),
                'processing_time': processing_time
            }
    
    def _save_results(self, results: Dict[str, Any]):
        """Зберегти результати"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"batch_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def get_processing_recommendations(self, tickers_count: int) -> Dict[str, Any]:
        """Отримати рекомендації для обробки"""
        recommendations = {
            'tickers_count': tickers_count,
            'recommended_strategy': 'adaptive',
            'recommended_batch_size': self.config.medium_batch_size,
            'estimated_time_minutes': 0,
            'memory_usage_gb': 0,
            'parallel_processing': True
        }
        
        # Розрахунок часу та пам'яті
        if tickers_count <= 10:
            recommendations['recommended_batch_size'] = self.config.small_batch_size
            recommendations['estimated_time_minutes'] = tickers_count * 2
            recommendations['memory_usage_gb'] = 2.0
        elif tickers_count <= 30:
            recommendations['recommended_batch_size'] = self.config.medium_batch_size
            recommendations['estimated_time_minutes'] = tickers_count * 1.5
            recommendations['memory_usage_gb'] = 4.0
        elif tickers_count <= 60:
            recommendations['recommended_batch_size'] = self.config.large_batch_size
            recommendations['estimated_time_minutes'] = tickers_count * 1.2
            recommendations['memory_usage_gb'] = 6.0
        else:
            recommendations['recommended_batch_size'] = self.config.max_batch_size
            recommendations['estimated_time_minutes'] = tickers_count * 1.0
            recommendations['memory_usage_gb'] = 8.0
            recommendations['recommended_strategy'] = 'priority'
        
        return recommendations


# Приклад використання
def example_processing_function(batch: List[str]) -> Dict[str, Any]:
    """Приклад функції обробки батчу"""
    # Симуляція обробки
    time.sleep(0.5)  # Симуляція роботи
    
    results = {}
    for ticker in batch:
        # Симуляція результатів
        results[ticker] = {
            'status': 'success',
            'data_points': np.random.randint(100, 1000),
            'accuracy': np.random.uniform(0.7, 0.9)
        }
    
    return results


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    # Створюємо процесор
    config = BatchConfig(
        strategy="adaptive",
        enable_parallel=True,
        max_workers=4
    )
    
    processor = BatchProcessor(config)
    
    # Отримуємо тікери
    tickers = get_tickers("tech")[:20]  # Перші 20 tech тікерів
    
    # Створюємо батчі
    batches = processor.create_optimal_batches(tickers)
    
    # Обробляємо
    results = processor.process_batches(batches, example_processing_function)
    
    print(f"Processing completed: {results['success_rate']:.2%} success rate")
    print(f"Total time: {results['processing_time']:.2f} seconds")
