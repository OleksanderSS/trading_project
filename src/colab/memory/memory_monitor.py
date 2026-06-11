"""Memory monitoring and management for Colab environment"""

import json
from datetime import datetime

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class MemoryMonitor:
    """Моніторинг та управління пам'яттю в Colab"""

    def __init__(self, warning_threshold=75, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_log = []
        self.start_time = datetime.now()

    def get_memory_info(self):
        """Отримати детальну інформацію про пам'ять"""
        import psutil
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'total_gb': mem.total / (1024**3)
        }

    def get_memory_usage(self):
        """Отримати відсоток використання пам'яті"""
        import psutil
        return psutil.virtual_memory().percent

    def check_memory(self, context=""):
        """Перевірити пам'ять та залогувати якщо потрібно"""
        info = self.get_memory_info()
        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'context': context,
            'memory_info': info
        }
        self.memory_log.append(log_entry)

        status = 'ok'
        if info['percent'] >= self.critical_threshold:
            logger.error(
                f"🚨 CRITICAL MEMORY: {info['percent']:.1f}% "
                f"({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)\n"
                f"   Context: {context}")
            status = 'critical'
        elif info['percent'] >= self.warning_threshold:
            logger.warning(
                f"⚠️ WARNING MEMORY: {info['percent']:.1f}% "
                f"({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)\n"
                f"   Context: {context}")
            status = 'warning'
        else:
            logger.info(
                f"✅ Memory OK: {info['percent']:.1f}% "
                f"({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")

        return status

    def cleanup(self):
        """Примусова збірка сміття"""
        import gc
        gc.collect()
        logger.info("🧹 Garbage collection triggered")

    def save_log(self, filepath):
        """Зберегти лог пам'яті у файл"""
        with open(filepath, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
        logger.info(f"💾 Memory log saved to {filepath}")
