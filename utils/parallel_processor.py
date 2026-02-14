#!/usr/bin/env python3
"""
Parallel Processor - Паралельна обробка data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Паралельна обробка data"""
    
    def __init__(self, config: Dict):
        self.max_workers = config.get('parallel', {}).get('max_workers', mp.cpu_count())
        self.use_processes = config.get('parallel', {}).get('use_processes', True)
        self.timeout = config.get('parallel', {}).get('timeout', 300)
        self.chunk_size = config.get('parallel', {}).get('chunk_size', 100)
        
    def process_tickers_parallel(self, 
                               tickers: List[str], 
                               timeframes: List[str], 
                               func: Callable,
                               **kwargs) -> List[Any]:
        """
        Паралельна обробка тікерів
        
        Args:
            tickers: Список тікерів
            timeframes: Список таймфреймів
            func: Функція для обробки
            **kwargs: Додаткові параметри
        
        Returns:
            List[Any]: Результати обробки
        """
        logger.info(f"[RESTART] Processing {len(tickers)} x {len(timeframes)} tasks in parallel")
        
        # [TARGET] Створюємо завдання
        tasks = []
        for ticker in tickers:
            for timeframe in timeframes:
                tasks.append((ticker, timeframe))
        
        logger.info(f"[DATA] Total tasks: {len(tasks)}, workers: {self.max_workers}, processes: {self.use_processes}")
        
        # [TARGET] Вибір виконавця
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = []
        start_time = time.time()
        
        with executor_class(max_workers=self.max_workers) as executor:
            # [TARGET] Запускаємо завдання
            future_to_task = {
                executor.submit(self._process_single_task, func, task, **kwargs): task
                for task in tasks
            }
            
            # [TARGET] Збираємо результати
            completed = 0
            for future in as_completed(future_to_task, timeout=self.timeout):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"[DATA] Completed {completed}/{len(tasks)} tasks in {elapsed:.1f}s")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Task {task} failed: {e}")
                    results.append(None)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r is not None)
        
        logger.info(f"[OK] Parallel processing completed: {successful}/{len(tasks)} successful in {elapsed:.1f}s")
        
        return results
    
    def _process_single_task(self, func: Callable, task: Tuple[str, str], **kwargs) -> Any:
        """Обробка одного завдання"""
        ticker, timeframe = task
        try:
            return func(ticker, timeframe, **kwargs)
        except Exception as e:
            logger.error(f"[ERROR] Error processing {ticker}_{timeframe}: {e}")
            raise
    
    def process_dataframe_chunks(self, 
                                 df: pd.DataFrame, 
                                 func: Callable,
                                 chunk_size: Optional[int] = None,
                                 **kwargs) -> pd.DataFrame:
        """
        Паралельна обробка DataFrame чанками
        
        Args:
            df: DataFrame для обробки
            func: Функція для обробки
            chunk_size: Розмір чанку
            **kwargs: Додаткові параметри
        
        Returns:
            pd.DataFrame: Об'єднаний результат
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        logger.info(f"[RESTART] Processing DataFrame in parallel chunks: {len(df)} rows, chunk_size: {chunk_size}")
        
        # [TARGET] Розбиваємо на чанки
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"[DATA] Created {len(chunks)} chunks")
        
        # [TARGET] Обробляємо чанки паралельно
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = []
        start_time = time.time()
        
        with executor_class(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(func, chunk, **kwargs): i
                for i, chunk in enumerate(chunks)
            }
            
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 5 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"[DATA] Completed {completed}/{len(chunks)} chunks in {elapsed:.1f}s")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Chunk {chunk_idx} failed: {e}")
                    results.append(pd.DataFrame())
        
        # [TARGET] Об'єднуємо результати
        if results:
            logger.info("[RESTART] Concatenating chunk results")
            final_result = pd.concat(results, ignore_index=True)
            
            # [TARGET] Сортуємо якщо є індекс
            if 'date' in final_result.columns:
                final_result = final_result.sort_values('date')
            
            return final_result
        else:
            return pd.DataFrame()
    
    def benchmark_performance(self, 
                            test_func: Callable,
                            test_data: List,
                            **kwargs) -> Dict[str, Any]:
        """
        Бенчмарк продуктивності
        
        Args:
            test_func: Тестова функція
            test_data: Тестові дані
            **kwargs: Додаткові параметри
        
        Returns:
            Dict[str, Any]: Результати бенчмарку
        """
        logger.info("🏃 Running performance benchmark...")
        
        # [TARGET] Послідовна обробка
        start_time = time.time()
        sequential_results = []
        for item in test_data:
            result = test_func(item, **kwargs)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # [TARGET] Паралельна обробка
        start_time = time.time()
        parallel_results = self.process_tickers_parallel(
            test_data, [], lambda x, y: test_func(x), **kwargs
        )
        parallel_time = time.time() - start_time
        
        # [TARGET] Розрахунок метрик
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / self.max_workers * 100
        
        benchmark_results = {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency_percent': efficiency,
            'workers': self.max_workers,
            'use_processes': self.use_processes,
            'items_processed': len(test_data),
            'items_per_second': len(test_data) / parallel_time
        }
        
        logger.info(f"[DATA] Benchmark results:")
        logger.info(f"   Sequential: {sequential_time:.2f}s")
        logger.info(f"   Parallel: {parallel_time:.2f}s")
        logger.info(f"   Speedup: {speedup:.2f}x")
        logger.info(f"   Efficiency: {efficiency:.1f}%")
        
        return benchmark_results
    
    def suggest_optimal_config(self, data_size: int, task_complexity: str = 'medium') -> Dict[str, Any]:
        """
        Підказати оптимальну конфігурацію
        
        Args:
            data_size: Розмір data
            task_complexity: Складність завдання ('low', 'medium', 'high')
        
        Returns:
            Dict[str, Any]: Рекомендації
        """
        recommendations = {}
        
        # [TARGET] Кількість workers
        if task_complexity == 'low':
            recommendations['max_workers'] = min(mp.cpu_count(), 8)
        elif task_complexity == 'medium':
            recommendations['max_workers'] = min(mp.cpu_count(), 4)
        else:  # high
            recommendations['max_workers'] = min(mp.cpu_count(), 2)
        
        # [TARGET] Processes vs Threads
        if task_complexity == 'high':
            recommendations['use_processes'] = True
        else:
            recommendations['use_processes'] = False
        
        # [TARGET] Chunk size
        if data_size < 1000:
            recommendations['chunk_size'] = 100
        elif data_size < 10000:
            recommendations['chunk_size'] = 500
        else:
            recommendations['chunk_size'] = 1000
        
        # [TARGET] Timeout
        if task_complexity == 'high':
            recommendations['timeout'] = 600
        else:
            recommendations['timeout'] = 300
        
        logger.info(f"[DATA] Recommended config for {task_complexity} complexity, {data_size} items:")
        for key, value in recommendations.items():
            logger.info(f"   {key}: {value}")
        
        return recommendations


if __name__ == "__main__":
    print("Parallel Processor - готовий до використання")
    print("[RESTART] Паралельна обробка data з оптимізацією")
