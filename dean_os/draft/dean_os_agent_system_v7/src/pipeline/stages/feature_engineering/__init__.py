"""Modular Stage 3 components with lazy imports."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FeatureEngineeringStage": (".orchestrator", "FeatureEngineeringStage"),
    "FeatureGuards": (".guards", "FeatureGuards"),
    "FeatureEnricher": (".enricher", "FeatureEnricher"),
    "TargetGenerator": (".targets", "TargetGenerator"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
