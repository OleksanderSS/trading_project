"""
PERFORMANCE MONITOR
Монandторинг продуктивностand system в реальному часand
"""

import psutil
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from config.pipeline_config import PERFORMANCE_CONFIG, LOGGING_CONFIG

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Монandторинг продуктивностand system
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.checkpoints = {}
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
        
    def start_checkpoint(self, name: str) -> None:
        """Почати checkpoint"""
        self.checkpoints[name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': self._get_cpu_usage()
        }
        self.logger.debug(f"[START] Started checkpoint: {name}")
    
    def end_checkpoint(self, name: str) -> Dict[str, Any]:
        """Закandнчити checkpoint"""
        if name not in self.checkpoints:
            self.logger.warning(f"[WARN] Checkpoint {name} not found")
            return {}
        
        checkpoint = self.checkpoints[name]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()
        
        metrics = {
            'name': name,
            'duration_seconds': end_time - checkpoint['start_time'],
            'memory_delta_mb': end_memory - checkpoint['start_memory'],
            'cpu_usage_avg': (checkpoint['start_cpu'] + end_cpu) / 2,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_history.append(metrics)
        self.logger.info(f"[OK] Completed checkpoint: {name} ({metrics['duration_seconds']:.2f}s, {metrics['memory_delta_mb']:.1f}MB)")
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Отримати поточнand метрики system"""
        return {
            'memory_usage_mb': self._get_memory_usage(),
            'cpu_usage_percent': self._get_cpu_usage(),
            'disk_usage_percent': self._get_disk_usage(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_memory_usage(self) -> float:
        """Отримати викорисandння пам'ятand в MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Отримати викорисandння CPU"""
        try:
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Отримати викорисandння диску"""
        try:
            return psutil.disk_usage('/').percent
        except:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Отримати withвandт про продуктивнandсть"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        total_duration = sum(m['duration_seconds'] for m in self.metrics_history)
        total_memory_delta = sum(m['memory_delta_mb'] for m in self.metrics_history)
        avg_cpu = sum(m['cpu_usage_avg'] for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            'total_duration_seconds': total_duration,
            'total_memory_delta_mb': total_memory_delta,
            'average_cpu_usage': avg_cpu,
            'checkpoints_completed': len(self.metrics_history),
            'slowest_checkpoint': max(self.metrics_history, key=lambda x: x['duration_seconds']),
            'memory_intensive_checkpoint': max(self.metrics_history, key=lambda x: x['memory_delta_mb']),
            'system_metrics': self.get_system_metrics()
        }
    
    def check_performance_limits(self) -> Dict[str, Any]:
        """Check лandмandти продуктивностand"""
        current_metrics = self.get_system_metrics()
        warnings = []
        
        # Перевandрка пам'ятand
        if current_metrics['memory_usage_mb'] > PERFORMANCE_CONFIG['memory_limit_gb'] * 1024:
            warnings.append(f"High memory usage: {current_metrics['memory_usage_mb']:.1f}MB")
        
        # Перевandрка CPU
        if current_metrics['cpu_usage_percent'] > 90:
            warnings.append(f"High CPU usage: {current_metrics['cpu_usage_percent']:.1f}%")
        
        # Перевandрка диску
        if current_metrics['disk_usage_percent'] > 90:
            warnings.append(f"High disk usage: {current_metrics['disk_usage_percent']:.1f}%")
        
        return {
            'status': 'warning' if warnings else 'ok',
            'warnings': warnings,
            'metrics': current_metrics
        }
    
    def log_performance_report(self) -> None:
        """Залогувати withвandт про продуктивнandсть"""
        if not LOGGING_CONFIG.get('performance_tracking', True):
            return
        
        summary = self.get_performance_summary()
        limits = self.check_performance_limits()
        
        self.logger.info("[DATA] PERFORMANCE REPORT")
        self.logger.info(f" Total duration: {summary.get('total_duration_seconds', 0):.2f}s")
        self.logger.info(f" Memory delta: {summary.get('total_memory_delta_mb', 0):.1f}MB")
        self.logger.info(f" Average CPU: {summary.get('average_cpu_usage', 0):.1f}%")
        self.logger.info(f"[UP] Checkpoints: {summary.get('checkpoints_completed', 0)}")
        
        if limits['warnings']:
            self.logger.warning("[WARN] PERFORMANCE WARNINGS:")
            for warning in limits['warnings']:
                self.logger.warning(f"   {warning}")


# Глобальний екwithемпляр монandтора
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Отримати глобальний монandтор продуктивностand"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def start_performance_checkpoint(name: str) -> None:
    """Почати checkpoint продуктивностand"""
    monitor = get_performance_monitor()
    monitor.start_checkpoint(name)

def end_performance_checkpoint(name: str) -> Dict[str, Any]:
    """Закandнчити checkpoint продуктивностand"""
    monitor = get_performance_monitor()
    return monitor.end_checkpoint(name)

def log_final_performance_report() -> None:
    """Залогувати фandнальний withвandт"""
    monitor = get_performance_monitor()
    monitor.log_performance_report()
