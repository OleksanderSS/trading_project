"""Colab training module with lazy public exports."""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "MemoryMonitor": ("src.colab.memory", "MemoryMonitor"),
    "get_optimal_batch_size": ("src.colab.utils", "get_optimal_batch_size"),
    "save_checkpoint": ("src.colab.utils", "save_checkpoint"),
    "load_checkpoint": ("src.colab.utils", "load_checkpoint"),
    "find_latest_checkpoint": ("src.colab.utils", "find_latest_checkpoint"),
    "retry_on_timeout": ("src.colab.utils", "retry_on_timeout"),
    "compute_data_signature": ("src.colab.utils", "compute_data_signature"),
    "compute_metrics": ("src.colab.utils", "compute_metrics"),
    "create_model": ("src.colab.models", "create_model"),
    "TrainingConfig": ("src.colab.config", "TrainingConfig"),
    "RuntimeConfigLoader": ("src.colab.config", "RuntimeConfigLoader"),
    "CheckpointParams": ("src.colab.config", "CheckpointParams"),
    "ColabEnvironment": ("src.colab.environment", "ColabEnvironment"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
