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
    Фоновий монітор ресурсів інфраструктури.
    Збирає детальні метрики CPU, пам'яті, диска та процесів.
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

        self.logger.info(f"ResourceMonitor ініціалізовано з порогами: {self.thresholds}")

    def _get_default_thresholds(self) -> Dict[str, float]:
        return {
            'cpu_warning': 70.0, 'cpu_critical': 90.0,
            'memory_warning': 80.0, 'memory_critical': 95.0,
            'disk_warning': 85.0, 'disk_critical': 95.0,
        }

    def start_monitoring(self, interval: int = 5):
        """Запускає фоновий потік моніторингу."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitoring_active = True
            self.monitor_thread = Thread(target=self._monitor_loop, args=(interval,), daemon=True)
            self.monitor_thread.start()
            self.logger.info(f"Моніторинг ресурсів запущено з інтервалом {interval}с.")

    def stop_monitoring(self):
        """Зупиняє фоновий потік моніторингу."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.executor.shutdown(wait=False)
        self.logger.info("Моніторинг ресурсів зупинено.")

    def _monitor_loop(self, interval: int):
        """Основний цикл моніторингу."""
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
                self.logger.error(f"Помилка в циклі моніторингу: {e}")

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Паралельно збирає всі системні метрики."""
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
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        top_cpu = sorted(processes, key=lambda p: p.get('cpu_percent', 0) or 0, reverse=True)[:5]
        top_mem = sorted(processes, key=lambda p: p.get('memory_percent', 0) or 0, reverse=True)[:5]

        return {'total': len(processes), 'top_cpu': top_cpu, 'top_memory': top_mem}

    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Перевіряє метрики на відповідність порогам і логує попередження."""
        cpu_percent = metrics.get('system', {}).get('cpu', {}).get('percent', 0)
        mem_percent = metrics.get('system', {}).get('memory', {}).get('percent', 0)
        disk_percent = metrics.get('disk', {}).get('usage', {}).get('percent', 0)
        
        if cpu_percent > self.thresholds['cpu_critical']:
            self.logger.critical(f"КРИТИЧНО: Використання CPU на рівні {cpu_percent:.1f}%")
        elif cpu_percent > self.thresholds['cpu_warning']:
            self.logger.warning(f"ПОПЕРЕДЖЕННЯ: Використання CPU на рівні {cpu_percent:.1f}%")

        if mem_percent > self.thresholds['memory_critical']:
            self.logger.critical(f"КРИТИЧНО: Використання пам'яті на рівні {mem_percent:.1f}%")
        elif mem_percent > self.thresholds['memory_warning']:
            self.logger.warning(f"ПОПЕРЕДЖЕННЯ: Використання пам'яті на рівні {mem_percent:.1f}%")

        if disk_percent > self.thresholds['disk_critical']:
            self.logger.critical(f"КРИТИЧНО: Використання диска на рівні {disk_percent:.1f}%")
        elif disk_percent > self.thresholds['disk_warning']:
            self.logger.warning(f"ПОПЕРЕДЖЕННЯ: Використання диска на рівні {disk_percent:.1f}%")

    def get_health_status(self) -> Dict[str, Any]:
        """Returns the latest comprehensive health metrics from the monitoring history."""
        with self.lock:
            if not self.performance_history:
                # If no history, collect current metrics synchronously as a fallback
                self.logger.warning("No historical metrics found. Collecting current metrics on demand.")
                try:
                    return self.collect_all_metrics()
                except Exception as e:
                    self.logger.error(f"Failed to collect on-demand metrics: {e}")
                    return {'status': 'error', 'message': 'Failed to collect metrics'}
            
            latest_metrics = self.performance_history[-1]
        
        # Add an overall status for quick assessment
        cpu_percent = latest_metrics.get('system', {}).get('cpu', {}).get('percent', 0)
        mem_percent = latest_metrics.get('system', {}).get('memory', {}).get('percent', 0)
        disk_percent = latest_metrics.get('disk', {}).get('usage', {}).get('percent', 0)
        
        status = 'good'
        if (cpu_percent > self.thresholds['cpu_critical'] or 
            mem_percent > self.thresholds['memory_critical'] or 
            disk_percent > self.thresholds['disk_critical']):
            status = 'critical'
        elif (cpu_percent > self.thresholds['cpu_warning'] or 
              mem_percent > self.thresholds['memory_warning'] or 
              disk_percent > self.thresholds['disk_warning']):
            status = 'warning'
            
        latest_metrics['overall_status'] = status
        return latest_metrics

# --- Singleton Instance ---
_resource_monitor_instance: Optional[ResourceMonitor] = None

def get_resource_monitor(config: Optional[Dict[str, Any]] = None) -> ResourceMonitor:
    """Отримати глобальний екземпляр ResourceMonitor."""
    global _resource_monitor_instance
    if _resource_monitor_instance is None:
        _resource_monitor_instance = ResourceMonitor(config)
    return _resource_monitor_instance

# --- Decorators ---
def track_resource_usage(monitor: ResourceMonitor = get_resource_monitor()):
    """Декоратор для відстеження часу виконання та успішності функції."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.monotonic() - start_time) * 1000  # у мс
                ProjectLogger.get_logger("ResourceTracker").info(
                    f"Функція '{func.__name__}' виконана за {duration:.2f}мс."
                )
        return wrapper
    return decorator