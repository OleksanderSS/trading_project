"""
Моніторинг продуктивності
"""

import time
from typing import Any

import numpy as np


class PerformanceMonitor:
    """Моніторинг продуктивності"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, name: str):
        """Початок таймера"""
        self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """Кінець таймера"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            return duration
        return 0.0

    def get_average_time(self, name: str) -> float:
        """Отримати середній час"""
        if name in self.metrics and self.metrics[name]:
            return np.mean(self.metrics[name])
        return 0.0

    def log_metric(self, name: str, value: Any):
        """Логування метрики"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Отримати підсумок метрик"""
        summary = {}

        for name, values in self.metrics.items():
            if isinstance(values[0], (int, float)):
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        return summary
