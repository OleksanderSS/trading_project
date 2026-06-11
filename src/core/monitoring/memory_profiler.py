"""
MemoryProfiler: Monitor memory usage and cleanup between pipeline stages

Tracks memory consumption during pipeline execution and provides automatic
cleanup to prevent memory leaks. Includes profiling decorators and context
managers for detailed memory analysis.

Features:
- Memory usage tracking per operation/stage
- Automatic cleanup with configurable thresholds
- Memory leak detection and warnings
- Integration with pipeline orchestrator

Usage:
    profiler = MemoryProfiler(warn_threshold_gb=8)

    with profiler.track("stage_3"):
        features = stage_3.run(...)

    # Automatic cleanup and logging
    profiler.cleanup()
"""
import gc
import logging
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import psutil


class MemoryProfiler:
    """
    Monitors memory usage during pipeline execution and provides cleanup utilities.

    Helps prevent memory leaks by tracking usage patterns and providing automatic
    garbage collection when thresholds are exceeded.

    Attributes:
        warn_threshold_gb: Memory usage threshold that triggers warnings
        critical_threshold_gb: Memory usage that triggers automatic cleanup
        process: psutil Process instance for memory monitoring
        _lock: Thread safety
    """

    def __init__(self, warn_threshold_gb: float=10.0, critical_threshold_gb:
        float=12.0):
        """
        Initialize memory profiler.

        Args:
            warn_threshold_gb: Memory usage (GB) that triggers warnings
            critical_threshold_gb: Memory usage (GB) that triggers automatic cleanup
        """
        self.warn_threshold_gb = warn_threshold_gb
        self.critical_threshold_gb = critical_threshold_gb
        self.process = psutil.Process()
        self._lock = threading.RLock()
        self.stats = {'peak_memory_gb': 0.0, 'operations_tracked': 0,
            'warnings_issued': 0, 'cleanups_performed': 0,
            'memory_freed_mb': 0.0}
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def track(self, operation_name: str, log_start: bool=True):
        """
        Context manager to track memory usage for an operation.

        Automatically logs memory delta and warns if usage is high.

        Args:
            operation_name: Name of the operation being tracked
            log_start: Whether to log memory usage at start

        Example:
            profiler = MemoryProfiler()
            with profiler.track("feature_engineering"):
                features = stage_3.run(...)
            # Memory usage is automatically logged
        """
        with self._lock:
            start_memory = self._get_memory_gb()
            start_time = time.time()
            if log_start:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'🧠 {operation_name} start - Memory: {start_memory:.2f}GB')
            try:
                yield
            finally:
                end_memory = self._get_memory_gb()
                end_time = time.time()
                delta_memory = end_memory - start_memory
                duration = end_time - start_time
                if end_memory > self.stats['peak_memory_gb']:
                    self.stats['peak_memory_gb'] = end_memory
                self.stats['operations_tracked'] += 1
                if abs(delta_memory) > 0.1:
                    self.logger.info(
                        f"🧠 {operation_name} completed - Memory: {end_memory:.2f}GB ({'+' if delta_memory >= 0 else ''}{delta_memory:.2f}GB), Duration: {duration:.1f}s"
                        )
                if end_memory > self.critical_threshold_gb:
                    self.logger.warning(
                        f'🚨 CRITICAL: Memory usage {end_memory:.2f}GB exceeds critical threshold {self.critical_threshold_gb}GB - triggering cleanup'
                        )
                    freed = self._perform_cleanup()
                    self.stats['cleanups_performed'] += 1
                    self.stats['memory_freed_mb'] += freed
                elif end_memory > self.warn_threshold_gb:
                    self.logger.warning(
                        f'⚠️ HIGH: Memory usage {end_memory:.2f}GB exceeds warning threshold {self.warn_threshold_gb}GB'
                        )
                    self.stats['warnings_issued'] += 1

    def track_function(self, func: Callable) ->Callable:
        """
        Decorator to track memory usage of a function.

        Args:
            func: Function to decorate

        Returns:
            Decorated function with memory tracking

        Example:
            @profiler.track_function
            def expensive_operation():
                return compute_features()
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f'{func.__name__}'
            with self.track(operation_name):
                return func(*args, **kwargs)
        return wrapper

    def cleanup(self, force: bool=False) ->float:
        """
        Perform memory cleanup.

        Args:
            force: If True, always perform cleanup regardless of memory usage

        Returns:
            Memory freed in MB
        """
        with self._lock:
            current_memory = self._get_memory_gb()
            if force or current_memory > self.warn_threshold_gb:
                freed_mb = self._perform_cleanup()
                if freed_mb > 10:
                    self.logger.info(f'🧹 Memory cleanup freed {freed_mb:.1f}MB'
                        )
                self.stats['cleanups_performed'] += 1
                self.stats['memory_freed_mb'] += freed_mb
                return freed_mb
            return 0.0

    def get_stats(self) ->dict[str, Any]:
        """
        Get memory profiling statistics.

        Returns:
            Dict with profiling statistics
        """
        with self._lock:
            current_memory = self._get_memory_gb()
            return {'current_memory_gb': current_memory, 'peak_memory_gb':
                self.stats['peak_memory_gb'], 'operations_tracked': self.
                stats['operations_tracked'], 'warnings_issued': self.stats[
                'warnings_issued'], 'cleanups_performed': self.stats[
                'cleanups_performed'], 'memory_freed_mb': self.stats[
                'memory_freed_mb'], 'warn_threshold_gb': self.
                warn_threshold_gb, 'critical_threshold_gb': self.
                critical_threshold_gb}

    def reset_stats(self) ->None:
        """Reset profiling statistics."""
        with self._lock:
            self.stats = {'peak_memory_gb': 0.0, 'operations_tracked': 0,
                'warnings_issued': 0, 'cleanups_performed': 0,
                'memory_freed_mb': 0.0}

    def _get_memory_gb(self) ->float:
        """Get current memory usage in GB."""
        try:
            memory_bytes = float(self.process.memory_info().rss)
            return float(memory_bytes / 1024 ** 3)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error getting memory usage: {e}', exc_info=True)
            return 0.0

    def _perform_cleanup(self) ->float:
        """
        Perform garbage collection and return memory freed.

        Returns:
            Memory freed in MB
        """
        try:
            before = self.process.memory_info().rss
            gc.collect()
            after = self.process.memory_info().rss
            freed_bytes = float(before - after)
            freed_mb = float(freed_bytes / 1024 ** 2)
            return float(max(0.0, freed_mb))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Memory cleanup failed: {e}', exc_info=True)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Memory cleanup measurement failed: {e}')
            return 0.0


_profiler: MemoryProfiler | None = None
_profiler_lock = threading.Lock()


def get_memory_profiler(warn_threshold_gb: float=10.0) ->MemoryProfiler:
    """
    Get or create global memory profiler (singleton).

    Args:
        warn_threshold_gb: Only used on first call to create profiler

    Returns:
        Global MemoryProfiler instance
    """
    global _profiler
    if _profiler is None:
        with _profiler_lock:
            if _profiler is None:
                _profiler = MemoryProfiler(warn_threshold_gb=warn_threshold_gb)
    return _profiler


def cleanup_memory() ->float:
    """Perform global memory cleanup."""
    profiler = get_memory_profiler()
    return profiler.cleanup(force=True)


def get_memory_stats() ->dict[str, Any]:
    """Get statistics from global profiler."""
    profiler = get_memory_profiler()
    return profiler.get_stats()
