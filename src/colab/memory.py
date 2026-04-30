"""Memory monitoring and management for Colab."""

import json
from datetime import datetime
from pathlib import Path


class MemoryMonitor:
    """Memory monitoring and management in Colab."""

    def __init__(self, warning_threshold=75, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_log = []
        self.start_time = datetime.now()

    def get_memory_info(self):
        """Get detailed memory information."""
        import psutil
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'total_gb': mem.total / (1024**3)
        }

    def get_memory_usage(self):
        """Get memory usage percentage."""
        import psutil
        return psutil.virtual_memory().percent

    def check_memory(self, context=""):
        """Check memory and log if needed."""
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
            print(
                f"🚨 CRITICAL MEMORY: {info['percent']:.1f}% "
                f"({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")
            print(f"   Context: {context}")
            status = 'critical'
        elif info['percent'] >= self.warning_threshold:
            print(
                f"⚠️ WARNING MEMORY: {info['percent']:.1f}% "
                f"({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")
            print(f"   Context: {context}")
            status = 'warning'
        else:
            print(
                f"✅ Memory OK: {info['percent']:.1f}% "
                f"({info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB)")

        return status

    def cleanup(self):
        """Force garbage collection."""
        import gc
        gc.collect()
        print("🧹 Garbage collection triggered")

    def save_log(self, filepath):
        """Save memory log to file."""
        with open(filepath, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
        print(f"💾 Memory log saved to {filepath}")


def get_optimal_batch_size(memory_percent, base_batch_size=32):
    """
    Calculate optimal batch size based on available memory.

    Logic:
    - If memory < 50%: use base_batch_size
    - If memory 50-75%: reduce to base_batch_size // 2
    - If memory 75-90%: reduce to base_batch_size // 4
    - If memory > 90%: reduce to base_batch_size // 8 (minimum 2)
    """
    if memory_percent < 50:
        return base_batch_size
    elif memory_percent < 75:
        return max(base_batch_size // 2, 8)
    elif memory_percent < 90:
        return max(base_batch_size // 4, 4)
    else:
        return max(base_batch_size // 8, 2)
