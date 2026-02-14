# utils/system_monitor.py

import os
import time
import gc
import psutil
from typing import Optional, Callable
from utils.logger import ProjectLogger
from config.system_config import SYSTEM_LIMITS

logger = ProjectLogger.get_logger("TradingProjectLogger")

def format_bytes(size_bytes: int) -> str:
    """Форматування пам'ятand в withроwithумandлand юнandти"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_resources() -> dict:
    """Отримати ресурси процесу."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        disk = psutil.disk_usage('/')
        net = psutil.net_io_counters()
        return {
            "rss": format_bytes(mem_info.rss),
            "rss_raw": mem_info.rss,
            "vms": format_bytes(mem_info.vms),
            "cpu": cpu_percent,
            "cpu_per_core": cpu_per_core,
            "disk_used": format_bytes(disk.used),
            "disk_free": format_bytes(disk.free),
            "net_sent": format_bytes(net.bytes_sent),
            "net_recv": format_bytes(net.bytes_recv),
        }
    except Exception as e:
        logger.error(f"[system_monitor] [ERROR] Error отримання ресурсandв: {e}")
        return {"rss": "0 B", "cpu": 0, "cpu_per_core": []}

def print_resources(verbose: bool = True) -> None:
    res = get_resources()
    if verbose and res.get("cpu_per_core"):
        cores_info = ', '.join(f"{c:.1f}%" for c in res['cpu_per_core'])
        logger.info(
            f" Пам'ять: RSS={res['rss']}, VMS={res['vms']} | "
            f"CPU={res['cpu']:.1f}% (по ядрах: {cores_info}) | "
            f"Диск: used={res['disk_used']}, free={res['disk_free']} | "
            f"Net: sent={res['net_sent']}, recv={res['net_recv']}"
        )

def check_thresholds(res: dict,
                     cpu_limit: int = SYSTEM_LIMITS["cpu_limit"],
                     mem_limit_gb: int = SYSTEM_LIMITS["mem_limit_gb"]):
    """Алерти при перевищеннand порогandв"""
    rss_mb = res.get("rss_raw", 0) / (1024 * 1024)
    if res.get("cpu", 0) > cpu_limit or rss_mb > mem_limit_gb * 1024:
        logger.warning(f"[system_monitor] [WARN] Перевищено порandг: CPU={res.get('cpu')}%, RSS={res.get('rss')}")

def log_run_metrics(start_time: float, task_name: str = "Завдання",
                    auto_gc: bool = False, verbose_resources: bool = True) -> None:
    duration = time.time() - start_time
    logger.info(f"[system_monitor] [DATA] Метрики '{task_name}': Час виконання={duration:.4f} сек")

    if auto_gc:
        before = gc.get_count()
        gc.collect()
        after = gc.get_count()
        logger.info(f"[system_monitor]  GC: before={before}, after={after}")
    if verbose_resources:
        res = get_resources()
        print_resources()
        check_thresholds(res)

class SystemMonitor:
    """Контекстний меnotджер for монandторингу"""
    def __init__(self, task_name: str, auto_gc: bool = False, verbose_resources: bool = True):
        self.task_name = task_name
        self.start_time: Optional[float] = None
        self.auto_gc = auto_gc
        self.verbose_resources = verbose_resources

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"[system_monitor] [START] Початок '{self.task_name}'...")
        if self.verbose_resources:
            print_resources()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - (self.start_time or time.time())
        if exc_type:
            logger.error(f"[system_monitor] [ERROR] Exception у '{self.task_name}': {exc_val}",
                exc_info=(exc_type,
                exc_val,
                exc_tb))
        logger.info(f"[system_monitor] [OK] Завершення '{self.task_name}': Час={duration:.4f} сек")
        log_run_metrics(self.start_time, self.task_name, auto_gc=self.auto_gc, verbose_resources=self.verbose_resources)

def monitor_task(task_name: str, auto_gc: bool = False, verbose_resources: bool = True):
    """Декоратор for монandторингу функцandй"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[system_monitor] [START] Початок '{task_name}'...")
            try:
                return func(*args, **kwargs)
            finally:
                log_run_metrics(start, task_name, auto_gc=auto_gc, verbose_resources=verbose_resources)
        return wrapper
    return decorator