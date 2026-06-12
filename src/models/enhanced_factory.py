"""
Enhanced Model Factory with Prototype Support

Extends ModelFactory with prototype pattern for fast model cloning and metadata management.
"""
import logging
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.factories.model_factory import ModelFactory
from src.models.interfaces import BaseModel
from src.models.prototypes.prototype import ModelPrototype
from src.models.prototypes.registry import PrototypeRegistry

logger = ProjectLogger.get_logger(__name__)


class EnhancedModelFactory(ModelFactory):
    """
    Enhanced factory with prototype pattern support.

    Features:
    - Backward compatible with ModelFactory
    - Prototype-based model creation
    - Dependency validation
    - Version tracking
    - Metadata management

    Example:
        factory = EnhancedModelFactory()

        # Register prototypes (one-time setup)
        factory.register_prototype(catboost_prototype)

        # Get model via prototype
        model = factory.get_model("catboost_v1", iterations=200)

        # Get model via legacy factory
        model = factory.get_model("catboost", iterations=100)
    """

    _PROTOTYPES: dict[str, ModelPrototype] = {}
    _REGISTRY: PrototypeRegistry | None = None

    @classmethod
    def initialize_registry(cls, registry_path: str = "data/prototypes/registry.json"):
        """
        Initialize prototype registry.

        Args:
            registry_path: Path to registry JSON file
        """
        cls._REGISTRY = PrototypeRegistry(registry_path=registry_path)
        logger.info(f"✅ Prototype registry initialized: {registry_path}")

    @classmethod
    def register_prototype(cls, prototype: ModelPrototype) -> bool:
        """
        Register a model prototype.

        Args:
            prototype: ModelPrototype to register

        Returns:
            True if registered successfully
        """
        cls._PROTOTYPES[prototype.model_id] = prototype

        if cls._REGISTRY:
            cls._REGISTRY.register(prototype)

        logger.info(f"✅ Registered prototype: {prototype.model_id}")
        return True

    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> BaseModel | None:
        """
        Get model instance via prototype or legacy factory.

        Tries prototype first, falls back to legacy factory.

        Args:
            model_name: Model identifier (e.g., 'catboost_v1' or 'catboost')
            **kwargs: Configuration parameters

        Returns:
            Model instance or None if not found/dependencies missing
        """
        # Try prototype first
        if model_name in cls._PROTOTYPES:
            prototype = cls._PROTOTYPES[model_name]
            model = prototype.clone(**kwargs)
            if model:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"✅ Got model from prototype: {model_name}")
                return model

        # Try registry
        if cls._REGISTRY:
            registry_prototype: ModelPrototype | None = cls._REGISTRY.get(model_name)
            if registry_prototype:
                model = registry_prototype.clone(**kwargs)
                if model:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"✅ Got model from registry: {model_name}")
                    return model

        # Fall back to legacy factory
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Falling back to legacy factory for: {model_name}")
        return super().get_model(model_name, **kwargs)

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of all available models (prototypes + legacy).

        Returns:
            List of model identifiers
        """
        legacy_models = super().get_available_models()
        prototype_models = list(cls._PROTOTYPES.keys())

        if cls._REGISTRY:
            registry_models = cls._REGISTRY.list_all()
            prototype_models.extend(registry_models)

        # Remove duplicates
        all_models = list(set(legacy_models + prototype_models))
        return sorted(all_models)

    @classmethod
    def get_prototype(cls, model_id: str) -> ModelPrototype | None:
        """
        Get prototype by ID.

        Args:
            model_id: Prototype identifier

        Returns:
            ModelPrototype or None if not found
        """
        # Check local prototypes
        if model_id in cls._PROTOTYPES:
            return cls._PROTOTYPES[model_id]

        # Check registry
        if cls._REGISTRY:
            return cls._REGISTRY.get(model_id)

        return None

    @classmethod
    def get_prototypes_by_type(cls, model_type: str) -> list:
        """
        Get all prototypes of specific type.

        Args:
            model_type: Model type to filter by

        Returns:
            List of matching prototypes
        """
        matching = []

        # Check local prototypes
        for proto in cls._PROTOTYPES.values():
            if model_type.lower() in proto.model_id.lower():
                matching.append(proto)

        # Check registry
        if cls._REGISTRY:
            registry_matching = cls._REGISTRY.get_by_type(model_type)
            matching.extend(registry_matching)

        return matching

    @classmethod
    def get_factory_stats(cls) -> dict[str, Any]:
        """
        Get factory statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "local_prototypes": len(cls._PROTOTYPES),
            "registry_initialized": cls._REGISTRY is not None,
        }

        if cls._REGISTRY:
            registry_stats = cls._REGISTRY.get_stats()
            stats.update({
                "registry_prototypes": registry_stats["total_prototypes"],
                "total_clones": registry_stats["total_clones"],
                "avg_clones_per_prototype": registry_stats["avg_clones_per_prototype"],
            })

        return stats

    @classmethod
    def export_summary(cls) -> dict[str, Any]:
        """
        Export factory summary.

        Returns:
            Dictionary with factory information
        """
        summary = {
            "local_prototypes": list(cls._PROTOTYPES.keys()),
            "available_models": cls.get_available_models(),
            "stats": cls.get_factory_stats(),
        }

        if cls._REGISTRY:
            summary["registry_summary"] = cls._REGISTRY.export_summary()

        return summary
