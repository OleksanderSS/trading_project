"""
Model Prototypes Module

Provides prototype pattern implementation for fast model cloning and metadata management.
"""

from src.models.prototypes.prototype import ModelPrototype
from src.models.prototypes.registry import PrototypeRegistry, get_prototype_registry

__all__ = [
    "ModelPrototype",
    "PrototypeRegistry",
    "get_prototype_registry",
]
