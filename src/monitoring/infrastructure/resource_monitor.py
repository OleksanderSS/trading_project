# src/monitoring/infrastructure/resource_monitor.py

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from src.core.logging.logger import ProjectLogger

class ResourceMonitor:
    """
    Background infrastructure resource monitor.
    Collects detailed CPU, memory, disk, and process metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_history: int = 1000):
        self.logger = ProjectLogger.get_logger("ResourceMonitor")
        self.thresholds = self._get_default_thresholds()
        if config and isinstance(config.get('thresholds'), dict):
            self.thresholds.update(config['thresholds'])
        
        self.monitoring_active = False
        self.monitor_thread: Optional[Thread] = None
        self.metrics_queue = queue.Queue()
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='ResMonitor')
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = max_history

        self.logger.info(f"ResourceMonitor initialized with thresholds: {self.thresholds}")

    def _get_default_thresholds(self) -> Dict[str, float]:
        return {
            'cpu_warning': 70.0, 'cpu_critical': 90.0,
            'memory_warning': 80.0, 'memory_critical': 95.0,
            'disk_warning': 85.0, 'disk_critical': 95.0,
        }

    def start_monitoring(self, interval: int = 5):
        """Starts background monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitoring_active = True
            self.monitor_thread = Thread(target=self._monitor_loop, args=(interval,), daemon=True)
            self.monitor_thread.start()
            self.logger.info(f"Resource monitoring started with interval {interval}с.")

    def stop_monitoring(self):
        """Stops background monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.executor.shutdown(wait=False)
        self.logger.info("Resource monitoring stopped.")

    def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                current_metrics = self.collect_all_metrics()
                self.metrics_queue.put(current_metrics)
                self._check_thresholds(current_metrics)
                
                with self.lock:
                    self.performance_history.append(current_metrics)
                    if len(self.performance_history) > self.max_history_size:
                        self.performance_history.pop(0)
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"error in monitoring loop: {e}")

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collects all system metrics in parallel."""
        futures = {
            'system': self.executor.submit(self._collect_system_metrics),
            'disk': self.executor.submit(self._collect_disk_metrics),
            'processes': self.executor.submit(self._collect_process_metrics)
        }
        return {
            'timestamp': datetime.now().isoformat(),
            'system': futures['system'].result(),
            'disk': futures['disk'].result(),
            'processes': futures['processes'].result(),
        }

    def _collect_system_metrics(self) -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'load_avg': psutil.getloadavg(),
            },
            'memory': {
                'percent': memory.percent,
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'swap_percent': swap.percent,
            }
        }

    def _collect_disk_metrics(self) -> Dict[str, Any]:
        disk_io = psutil.disk_io_counters()
        disk_usage = psutil.disk_usage('/')
        return {
            'io': {
                'read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
            },
            'usage': {
                'percent': disk_usage.percent,
                'free_gb': disk_usage.free / (1024**3),
                'total_gb': disk_usage.total / (1024**3),
            }
        }

    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process metrics with timeout to prevent hanging on Windows."""
        try:
            processes = []
            # Limit to first 100 processes to avoid timeout on Windows
            count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                    count += 1
                    if count >= 100:  # Limit collection
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
            
            top_cpu = sorted(processes, key=lambda p: p.get('cpu_percent', 0) or 0, reverse=True)[:5]
            top_mem = sorted(processes, key=lambda p: p.get('memory_percent', 0) or 0, reverse=True)[:5]

            return {'total': len(processes), 'top_cpu': top_cpu, 'top_memory': top_mem}
        except Exception as e:
            # Fallback if process collection fails
            return {'total': 0, 'top_cpu': [], 'top_memory': [], 'error': str(e)}

    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Checks metrics against thresholds and logs warnings."""
        cpu_percent = metrics.get('system', {}).get('cpu', {}).get('percent', 0)
        mem_percent = metrics.get('system', {}).get('memory', {}).get('percent', 0)
        disk_percent = metrics.get('disk', {}).get('usage', {}).get('percent', 0)
        
        if cpu_percent > self.thresholds['cpu_critical']:
            self.logger.critical(f"CRITICAL: CPU usage at {cpu_percent:.1f}%")
        elif cpu_percent > self.thresholds['cpu_warning']:
            self.logger.warning(f"WARNING: CPU usage at {cpu_percent:.1f}%")

        if mem_percent > self.thresholds['memory_critical']:
            self.logger.critical(f"CRITICAL: Memory usage at {mem_percent:.1f}%")
        elif mem_percent > self.thresholds['memory_warning']:
            self.logger.warning(f"WARNING: Memory usage at {mem_percent:.1f}%")

        if disk_percent > self.thresholds['disk_critical']:
            self.logger.critical(f"CRITICAL: Disk usage at {disk_percent:.1f}%")
        elif disk_percent > self.thresholds['disk_warning']:
            self.logger.warning(f"WARNING: Disk usage at {disk_percent:.1f}%")

    def get_health_status(self) -> Dict[str, Any]:
        """Returns the latest comprehensive health metrics from the monitoring history."""
        latest_metrics = self._get_latest_metrics_or_collect()
        return self._format_health_response(latest_metrics)

    def _format_health_response(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format health response based on metrics status."""
        if self._is_error_status(metrics):
            return metrics
        return self._prepare_health_report(metrics)

    def _is_error_status(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics indicate an error status."""
        return metrics.get('status') == 'error'
    
    def _prepare_health_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final health report with overall status."""
        status = self._determine_overall_status(metrics)
        metrics['overall_status'] = status
        return metrics
    
    def _get_latest_metrics_or_collect(self) -> Dict[str, Any]:
        """Gets latest metrics or collects on-demand if no history."""
        with self.lock:
            if not self.performance_history:
                return self._collect_fallback_metrics()
            return self.performance_history[-1]

    def _collect_fallback_metrics(self) -> Dict[str, Any]:
        """Collects metrics synchronously when no history is available."""
        self.logger.warning("No historical metrics found. Collecting current metrics on demand.")
        try:
            return self.collect_all_metrics()
        except Exception as e:
            self.logger.error(f"Failed to collect on-demand metrics: {e}")
            return {'status': 'error', 'message': 'Failed to collect metrics'}

    def _determine_overall_status(self, metrics: Dict[str, Any]) -> str:
        """Determine overall system status based on metrics."""
        resource_levels = self._extract_resource_levels(metrics)
        
        if self._is_critical_level(**resource_levels):
            return 'critical'
        elif self._is_warning_level(**resource_levels):
            return 'warning'
        return 'good'

    def _extract_resource_levels(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract CPU, memory, and disk percentages from metrics."""
        return {
            'cpu_percent': self._get_cpu_percent(metrics),
            'mem_percent': self._get_memory_percent(metrics),
            'disk_percent': self._get_disk_percent(metrics)
        }

    def _get_cpu_percent(self, metrics: Dict[str, Any]) -> float:
        """Extract CPU percentage from metrics."""
        return self._safe_nested_get(metrics, ['system', 'cpu', 'percent'], 0)

    def _get_memory_percent(self, metrics: Dict[str, Any]) -> float:
        """Extract memory percentage from metrics."""
        return self._safe_nested_get(metrics, ['system', 'memory', 'percent'], 0)

    def _get_disk_percent(self, metrics: Dict[str, Any]) -> float:
        """Extract disk percentage from metrics."""
        return self._safe_nested_get(metrics, ['disk', 'usage', 'percent'], 0)

    def _safe_nested_get(self, data: Dict[str, Any], keys: list, default: Any) -> Any:
        """Safely get nested dictionary value with fallback."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _is_critical_level(self, cpu_percent: float, mem_percent: float, disk_percent: float) -> bool:
        """Check if any resource is at critical level."""
        return (cpu_percent > self.thresholds['cpu_critical'] or
                mem_percent > self.thresholds['memory_critical'] or
                disk_percent > self.thresholds['disk_critical'])

    def _is_warning_level(self, cpu_percent: float, mem_percent: float, disk_percent: float) -> bool:
        """Check if any resource is at warning level."""
        return (cpu_percent > self.thresholds['cpu_warning'] or
                mem_percent > self.thresholds['memory_warning'] or
                disk_percent > self.thresholds['disk_warning'])

# --- Singleton Instance ---
_resource_monitor_instance: Optional[ResourceMonitor] = None

def get_resource_monitor(config: Optional[Dict[str, Any]] = None) -> ResourceMonitor:
    """Get global instance ResourceMonitor."""
    global _resource_monitor_instance
    if _resource_monitor_instance is None:
        _resource_monitor_instance = ResourceMonitor(config)
    return _resource_monitor_instance

# --- Decorators ---
def track_resource_usage():
    """Decorator to track execution time and successfully ."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.monotonic() - start_time) * 1000  # in ms
                ProjectLogger.get_logger("ResourceTracker").info(
                    f"Функція '{func.__name__}' виконана за {duration:.2f}мс."
                )
        return wrapper
    return decorator