import logging
# src/core/processing/parallel_processor.py

import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional, Tuple

import pandas as pd
import psutil

from src.core.logging.logger import ProjectLogger

# Initialize logger for the module
logger = ProjectLogger.get_logger("ParallelProcessor")

class ParallelProcessor:
    """Provides centralized parallel processing capabilities."""

    def __init__(self, max_workers: int = None, use_processes: bool = False, timeout: int = 300, chunk_size: int = 100):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.logger = logger
        
        # Initial memory check
        mem = psutil.virtual_memory()
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"ParallelProcessor initialized. System RAM: {mem.total / (1024**3):.2f} GB ({mem.percent}% used)")

    def _get_executor(self):
        """Returns the appropriate executor based on the configuration."""
        if self.use_processes:
            return ProcessPoolExecutor(max_workers=self.max_workers)
        return ThreadPoolExecutor(max_workers=self.max_workers)

    def process_items(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """
        Processes a list of items in parallel.

        Args:
            func: The function to apply to each item.
            items: A list of items to process.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A list of results.
        """
        if not items:
            return []

        self.logger.info(f"Processing {len(items)} items in parallel (workers={self.max_workers}, processes={self.use_processes})")
        results = []
        start_time = time.time()

        with self._get_executor() as executor:
            future_to_item = {executor.submit(func, item, **kwargs): i for i, item in enumerate(items)}

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing item #{future_to_item[future]}: {e}", exc_info=True)
                    results.append(None) # Append None to maintain order

        elapsed = time.time() - start_time
        self.logger.info(f"Parallel processing of {len(items)} items completed in {elapsed:.2f}s")
        return results

    def process_dataframe_chunks(self, df: pd.DataFrame, func: Callable, chunk_size: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Processes a DataFrame in parallel by splitting it into chunks.

        Args:
            df: The DataFrame to process.
            func: The function to apply to each chunk.
            chunk_size: The size of each chunk. Uses instance default if None.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A new DataFrame with the concatenated results.
        """
        base_chunk_size = chunk_size or self.chunk_size
        if df.empty or base_chunk_size <= 0:
            return pd.DataFrame()

        # Dynamic memory check and chunk size adjustment
        mem = psutil.virtual_memory()
        effective_chunk_size = base_chunk_size
        
        if mem.percent > 80:
            effective_chunk_size = max(1, base_chunk_size // 2)
            self.logger.warning(
                f"Memory critical ({mem.percent}% used). Reducing chunk_size from {base_chunk_size} to {effective_chunk_size}."
            )

        chunks = [df.iloc[i:i + effective_chunk_size] for i in range(0, len(df), effective_chunk_size)]
        self.logger.info(f"Processing DataFrame with {len(df)} rows in {len(chunks)} chunks.")

        # The processing function for chunks should be `func(chunk, **kwargs)`
        chunk_results = self.process_items(func, chunks, **kwargs)

        # Filter out None results from failed chunks and concatenate
        valid_results = [res for res in chunk_results if res is not None and not res.empty]
        if not valid_results:
            return pd.DataFrame()

        self.logger.info("Concatenating chunk results.")
        final_df = pd.concat(valid_results, ignore_index=True)
        
        # Optional: Sort if a date column exists to maintain order
        if 'date' in final_df.columns:
            final_df = final_df.sort_values('date').reset_index(drop=True)
            
        return final_df

    def process_tickers_parallel(self, func: Callable, tickers: List[str], timeframes: List[str], **kwargs) -> List[Any]:
        """
        Processes a combination of tickers and timeframes in parallel.
        This is a specialized method useful for financial data processing.

        Args:
            func: The function to apply. It must accept (ticker, timeframe, **kwargs).
            tickers: A list of tickers to process.
            timeframes: A list of timeframes to process for each ticker.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A list of results.
        """
        if not tickers or not timeframes:
            self.logger.warning("process_tickers_parallel called with empty tickers or timeframes. Returning empty list.")
            return []

        tasks = [(ticker, timeframe) for ticker in tickers for timeframe in timeframes]
        self.logger.info(f"Generated {len(tasks)} ticker/timeframe tasks for parallel processing.")

        # This wrapper function unpacks the tuple task before calling the user-provided function.
        # It allows us to use the generic `process_items` method for this specific task structure.
        def task_wrapper(task_tuple: Tuple[str, str], **inner_kwargs: Any) -> Any:
            ticker, timeframe = task_tuple
            return func(ticker, timeframe, **inner_kwargs)

        return self.process_items(task_wrapper, tasks, **kwargs)

    def benchmark(self, test_func: Callable, test_data: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Benchmarks sequential vs. parallel execution for a given function and data.

        Args:
            test_func: The function to benchmark.
            test_data: The data to process.
            **kwargs: Additional arguments for the function.

        Returns:
            A dictionary with benchmark results.
        """
        self.logger.info("Running performance benchmark...")

        # Sequential execution
        start_seq = time.time()
        for item in test_data:
            test_func(item, **kwargs)
        sequential_time = time.time() - start_seq

        # Parallel execution
        start_par = time.time()
        self.process_items(test_func, test_data, **kwargs)
        parallel_time = time.time() - start_par

        speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
        efficiency = (speedup / self.max_workers) * 100

        results = {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency_percent': efficiency,
            'workers': self.max_workers,
            'items_processed': len(test_data)
        }

        self.logger.info(f"Benchmark Results: Sequential={sequential_time:.2f}s, Parallel={parallel_time:.2f}s, Speedup={speedup:.2f}x")
        return results

    def suggest_optimal_config(self, data_size: int, task_complexity: str = 'medium') -> Dict[str, Any]:
        """
        Suggests an optimal configuration based on data size and task complexity.

        Args:
            data_size: The number of items to process.
            task_complexity: 'low' (I/O-bound), 'medium', or 'high' (CPU-bound).

        Returns:
            A dictionary with recommended settings.
        """
        cpu_count = mp.cpu_count()
        recommendations = {}

        # Recommend process-based for CPU-bound tasks, thread-based for I/O
        recommendations['use_processes'] = True if task_complexity == 'high' else False

        # Adjust worker count based on complexity
        if task_complexity == 'high': # CPU-bound, don't oversubscribe cores
            recommendations['max_workers'] = cpu_count
        elif task_complexity == 'low': # I/O-bound, can have more workers
            recommendations['max_workers'] = min(cpu_count * 2, 32)
        else: # Medium
            recommendations['max_workers'] = cpu_count

        # Adjust chunk size based on data size
        if data_size < 1000:
            recommendations['chunk_size'] = 100
        elif data_size < 100000:
            recommendations['chunk_size'] = 1000
        else:
            recommendations['chunk_size'] = 5000

        self.logger.info(f"Suggested config for {data_size} items ({task_complexity} complexity): {recommendations}")
        return recommendations