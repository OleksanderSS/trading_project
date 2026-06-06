"""
Prototype Registry for managing model prototypes

Provides centralized storage and retrieval of prototypes with persistence.
"""
import json
import logging
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.models.prototypes.prototype import ModelPrototype

logger = ProjectLogger.get_logger(__name__)


class PrototypeRegistry:
    """
    Central registry for model prototypes.

    Features:
    - Register/retrieve prototypes
    - Persistence to disk
    - Filtering by type/version
    - Validation
    - Statistics tracking

    Example:
        registry = PrototypeRegistry()
        registry.register(catboost_prototype)
        registry.register(lstm_prototype)

        model = registry.clone("catboost_v1", iterations=200)
        all_prototypes = registry.list_all()
    """

    def __init__(self, registry_path: str = "data/prototypes/registry.json"):
        """
        Initialize prototype registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.prototypes: dict[str, ModelPrototype] = {}
        self._load_registry()

    def register(self, prototype: ModelPrototype) -> bool:
        """
        Register a new prototype.

        Args:
            prototype: ModelPrototype to register

        Returns:
            True if registered successfully
        """
        if prototype.model_id in self.prototypes:
            logger.warning(
                f"Prototype {prototype.model_id} already registered. Overwriting."
            )

        self.prototypes[prototype.model_id] = prototype
        self._save_registry()
        logger.info(f"✅ Registered prototype: {prototype.model_id}")
        return True

    def get(self, model_id: str) -> ModelPrototype | None:
        """
        Get prototype by ID.

        Args:
            model_id: Prototype identifier

        Returns:
            ModelPrototype or None if not found
        """
        return self.prototypes.get(model_id)

    def clone(self, model_id: str, **kwargs):
        """
        Clone model from prototype.

        Args:
            model_id: Prototype identifier
            **kwargs: Parameters to override

        Returns:
            Model instance or None if prototype not found
        """
        prototype = self.get(model_id)
        if not prototype:
            logger.error(f"Prototype not found: {model_id}")
            return None
        return prototype.clone(**kwargs)

    def list_all(self) -> list[str]:
        """
        List all registered prototype IDs.

        Returns:
            List of prototype IDs
        """
        return list(self.prototypes.keys())

    def get_by_type(self, model_type: str) -> list[ModelPrototype]:
        """
        Get all prototypes of specific type.

        Args:
            model_type: Model type to filter by (case-insensitive)

        Returns:
            List of matching prototypes
        """
        return [
            p
            for p in self.prototypes.values()
            if model_type.lower() in p.model_id.lower()
        ]

    def get_by_version(self, model_id: str, version: str) -> ModelPrototype | None:
        """
        Get specific version of prototype.

        Args:
            model_id: Base model ID
            version: Version string

        Returns:
            ModelPrototype or None if not found
        """
        full_id = f"{model_id}_v{version}"
        return self.get(full_id)

    def remove(self, model_id: str) -> bool:
        """
        Remove prototype from registry.

        Args:
            model_id: Prototype identifier

        Returns:
            True if removed successfully
        """
        if model_id in self.prototypes:
            del self.prototypes[model_id]
            self._save_registry()
            logger.info(f"🗑️ Removed prototype: {model_id}")
            return True
        return False

    def _save_registry(self):
        """Save registry to disk"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            model_id: prototype.get_info()
            for model_id, prototype in self.prototypes.items()
        }

        try:
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"💾 Registry saved: {len(data)} prototypes")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to save registry: {e}")

    def _load_registry(self):
        """Load registry from disk"""
        if not self.registry_path.exists():
            logger.info("📝 No existing registry found. Starting fresh.")
            return

        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            # Reconstruct prototypes
            from src.models.prototypes.prototype import ModelPrototype

            # Note: We need a way to resolve model_class from name.
            # For now, if it's MockModel, we assume the test environment context.
            # In production, we'd use a registry or importlib.

            for model_id, info in data.items():
                proto = ModelPrototype(
                    model_id=model_id,
                    model_class=Any, # Simplified for registry
                    version=info["version"],
                    dependencies=info["dependencies"],
                    metadata=info["metadata"]
                )
                self.prototypes[model_id] = proto

            logger.info(f"📖 Loaded {len(data)} prototypes from registry")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to load registry: {e}")

    def export_summary(self) -> dict[str, Any]:
        """
        Export registry summary.

        Returns:
            Dictionary with registry information
        """
        return {
            "total_prototypes": len(self.prototypes),
            "prototypes": [p.get_info() for p in self.prototypes.values()],
            "registry_path": str(self.registry_path),
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        total_clones = sum(p._clone_count for p in self.prototypes.values())
        avg_clones = (
            total_clones / len(self.prototypes) if self.prototypes else 0
        )

        return {
            "total_prototypes": len(self.prototypes),
            "total_clones": total_clones,
            "avg_clones_per_prototype": avg_clones,
            "prototypes": list(self.prototypes.keys()),
        }

    def __repr__(self) -> str:
        return f"PrototypeRegistry(prototypes={len(self.prototypes)})"


# Global singleton
_registry: PrototypeRegistry | None = None


def get_prototype_registry() -> PrototypeRegistry:
    """
    Get or create global prototype registry.

    Returns:
        Global PrototypeRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PrototypeRegistry()
    return _registry
