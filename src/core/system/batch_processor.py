#!/usr/bin/env python3
"""
Batch Processing Manager for large ticker sets.
Optimized for rapid processing of 50+ tickers.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@dataclass
class BatchConfig:
    """Batch processing configuration"""
    # Batch sizes
    small_batch_size: int = 5
    medium_batch_size: int = 10
    large_batch_size: int = 20
    max_batch_size: int = 50
    
    # Resources
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    timeout_seconds: int = 300
    
    # Optimization
    enable_parallel: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    
    # Strategies
    strategy: str = "adaptive"
    priority_categories: List[str] = field(default_factory=lambda: ["tech", "finance", "core"])

class BatchProcessor:
    """Ticker batch processing manager"""
    
    def __init__(self, config: BatchConfig = None, cache_dir: Optional[Path] = None, results_dir: Optional[Path] = None):
        self.config = config or BatchConfig()
        self.logger = logging.getLogger("BatchProcessor")
        
        # Use provided paths or defaults
        self.cache_dir = cache_dir or Path("cache/batch")
        self.results_dir = results_dir or Path("results/batch")
        
        # Create directories if they do not exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.processing_stats = {
            'total_tickers': 0,
            'processed_tickers': 0,
            'failed_tickers': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None,
            'processing_time': 0.0
        }
        
        self._lock = threading.Lock()
        self.logger.info("BatchProcessor initialized")
    
    def create_optimal_batches(self, tickers: List[str], ticker_categories: Optional[Dict[str, List[str]]] = None) -> List[List[str]]:
        """
        Create optimal batches for processing.
        
        Args:
            tickers: List of tickers
            ticker_categories: Dictionary where key is ticker and value is list of its categories
            
        Returns:
            List[List[str]]: List of batches
        """
        if not tickers:
            return []
        
        self.logger.info(f"Creating optimal batches for {len(tickers)} tickers")
        
        if self.config.strategy == "adaptive":
            return self._create_adaptive_batches(tickers)
        elif self.config.strategy == "priority":
            if not ticker_categories:
                self.logger.warning("Priority strategy requires ticker_categories, but it was not provided. Falling back to adaptive strategy.")
                return self._create_adaptive_batches(tickers)
            return self._create_priority_batches(tickers, ticker_categories)
        else:
            return self._create_fixed_batches(tickers)

    def _create_adaptive_batches(self, tickers: List[str]) -> List[List[str]]:
        """Adaptive batch creation"""
        total_tickers = len(tickers)
        
        if total_tickers <= 10:
            batch_size = self.config.small_batch_size
        elif total_tickers <= 30:
            batch_size = self.config.medium_batch_size
        elif total_tickers <= 60:
            batch_size = self.config.large_batch_size
        else:
            batch_size = self.config.max_batch_size
        
        batches = [tickers[i:i + batch_size] for i in range(0, total_tickers, batch_size)]
        self.logger.info(f"Created {len(batches)} adaptive batches (size: {batch_size})")
        return batches

    def _create_priority_batches(self, tickers: List[str], ticker_categories: Dict[str, List[str]]) -> List[List[str]]:
        """Priority-based batch creation"""
        priority_tickers = []
        other_tickers = []
        
        for ticker in tickers:
            categories = ticker_categories.get(ticker, [])
            if any(cat in self.config.priority_categories for cat in categories):
                priority_tickers.append(ticker)
            else:
                other_tickers.append(ticker)
        
        all_tickers = priority_tickers + other_tickers
        return self._create_fixed_batches(all_tickers)

    def _create_fixed_batches(self, tickers: List[str]) -> List[List[str]]:
        """Fixed-size batch creation"""
        batch_size = self.config.medium_batch_size
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        self.logger.info(f"Created {len(batches)} fixed batches (size: {batch_size})")
        return batches

    def process_batches(self, batches: List[List[str]], processing_func: Callable[[List[str]], Any]) -> Dict[str, Any]:
        """
        Process batches.
        """
        if not batches:
            return {'status': 'error', 'message': 'No batches to process'}
        
        self.processing_stats['total_tickers'] = sum(len(batch) for batch in batches)
        self.processing_stats['start_time'] = time.time()
        self.logger.info(f"Starting batch processing: {len(batches)} batches, {self.processing_stats['total_tickers']} total tickers")
        
        results = self._process_batches_parallel(batches, processing_func) if self.config.enable_parallel and len(batches) > 1 else self._process_batches_sequential(batches, processing_func)
        
        self.processing_stats['end_time'] = time.time()
        self.processing_stats['processing_time'] = self.processing_stats['end_time'] - self.processing_stats['start_time']
        
        final_results = {
            'status': 'success',
            'batches_processed': len(results),
            'total_tickers': self.processing_stats['total_tickers'],
            'processed_tickers': self.processing_stats['processed_tickers'],
            'failed_tickers': self.processing_stats['failed_tickers'],
            'processing_time': self.processing_stats['processing_time'],
            'batch_results': results,
            'success_rate': (self.processing_stats['processed_tickers'] / self.processing_stats['total_tickers']) if self.processing_stats['total_tickers'] > 0 else 0
        }
        
        self.logger.info(f"Batch processing completed: {final_results['success_rate']:.2%} success rate")
        self._save_results(final_results)
        return final_results

    def _process_batches_parallel(self, batches: List[List[str]], processing_func: Callable[[List[str]], Any]) -> List[Dict[str, Any]]:
        """Parallel batch processing"""
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_batch = {executor.submit(self._process_single_batch, batch, processing_func): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    with self._lock:
                        self.processing_stats['processed_tickers'] += result.get('processed_count', 0)
                        self.processing_stats['failed_tickers'] += result.get('failed_count', 0)
                        self.processing_stats['batches_processed'] += 1
                    progress = (self.processing_stats['batches_processed'] / len(batches)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% - Batch {self.processing_stats['batches_processed']}/{len(batches)}")
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    results.append({'batch': batch, 'status': 'failed', 'error': str(e), 'processed_count': 0, 'failed_count': len(batch)})
        return results

    def _process_batches_sequential(self, batches: List[List[str]], processing_func: Callable[[List[str]], Any]) -> List[Dict[str, Any]]:
        """Sequential batch processing"""
        results = []
        for i, batch in enumerate(batches):
            try:
                result = self._process_single_batch(batch, processing_func)
                results.append(result)
                self.processing_stats['processed_tickers'] += result.get('processed_count', 0)
                self.processing_stats['failed_tickers'] += result.get('failed_count', 0)
                self.processing_stats['batches_processed'] += 1
                progress = ((i + 1) / len(batches)) * 100
                self.logger.info(f"Progress: {progress:.1f}% - Batch {i + 1}/{len(batches)}")
            except Exception as e:
                self.logger.error(f"Batch {i + 1} processing failed: {e}")
                results.append({'batch': batch, 'status': 'failed', 'error': str(e), 'processed_count': 0, 'failed_count': len(batch)})
        return results

    def _process_single_batch(self, batch: List[str], processing_func: Callable[[List[str]], Any]) -> Dict[str, Any]:
        """Processing of a single batch"""
        start_time = time.time()
        try:
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
        """Save results to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"batch_results_{timestamp}.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {results_file}")
