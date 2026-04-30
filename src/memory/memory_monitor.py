"""
Моніторинг та управління пам'яттю в Colab
"""
from datetime import datetime
from typing import Dict, Any
import psutil


class MemoryMonitor:
    """Моніторинг та управління пам'яттю"""

    def __init__(self, warning_threshold: int = 75, critical_threshold: int = 90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_log = []
        self.start_time = datetime.now()

    def get_memory_info(self) -> Dict[str, float]:
        """Отримати детальну інформацію про пам'ять"""
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'total_gb': mem.total / (1024**3)
        }

    def get_memory_usage(self) -> float:
        """Отримати відсоток використання пам'яті"""
        return psutil.virtual_memory().percent

    def check_memory(self, context: str = "") -> Dict[str, Any]:
        """Перевірити пам'ять та залогувати якщо потрібно"""
        info = self.get_memory_info()
        timestamp = datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'context': context,
            'memory_info': info
        }
        self.memory_log.append(log_entry)

        status = self._determine_status(info['percent'])
        
        if status in ['warning', 'critical']:
            print(f"[{status.upper()}] Пам'ять: {info['percent']:.1f}% | "
                  f"Використано: {info['used_gb']:.2f}GB / {info['total_gb']:.2f}GB")

        return {'status': status, 'info': info}

    def _determine_status(self, memory_percent: float) -> str:
        """Визначити статус пам'яті"""
        if memory_percent >= self.critical_threshold:
            return 'critical'
        elif memory_percent >= self.warning_threshold:
            return 'warning'
        return 'ok'

    def cleanup(self) -> None:
        """Очистити пам'ять"""
        import gc
        gc.collect()
        print("Пам'ять очищена")

    def save_log(self, filepath: str) -> None:
        """Зберегти лог пам'яті"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
