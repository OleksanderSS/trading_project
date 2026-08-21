# src/features/validation/__init__.py
#
# Exported lazily, because importing a submodule runs this file first.
#
# Callers reach for one thing or the other -- `colab_manager` and stage 3's
# guards import `feature_leakage_guard`, `enhanced_smart_selector` imports
# `redundancy_detector` -- but eagerly importing both here meant every one of
# them paid for `sklearn.cluster`, which `redundancy_detector` needs and the
# leakage guard does not.
#
# Measured with `python -X importtime run_hybrid_pipeline.py --help`, after
# Evidently was taken off the same path:
#
#     src.features.validation           8.6 s
#     sklearn.cluster                   8.6 s
#     sklearn                           7.6 s
#
# Eight and a half seconds of clustering machinery to print a help message.
# PEP 562 keeps `from src.features.validation import RedundancyDetector`
# working exactly as before; it just happens when someone asks.

from typing import Any

__all__ = [
    "FeatureLeakageGuard",
    "LeakageReport",
    "get_leakage_guard",
    "RedundancyDetector",
    "get_redundancy_detector",
    "eliminate_redundancy_quick",
]

_SOURCE = {
    "FeatureLeakageGuard": ".feature_leakage_guard",
    "LeakageReport": ".feature_leakage_guard",
    "get_leakage_guard": ".feature_leakage_guard",
    "RedundancyDetector": ".redundancy_detector",
    "get_redundancy_detector": ".redundancy_detector",
    "eliminate_redundancy_quick": ".redundancy_detector",
}


def __getattr__(name: str) -> Any:
    module_name = _SOURCE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cached, so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
