"""
Model Prototype Pattern Implementation

Provides fast cloning and metadata management for models.
"""
import importlib
import logging
from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.models.interfaces import BaseModel

logger = ProjectLogger.get_logger(__name__)


class ModelPrototype:
    """
    Prototype pattern for models with metadata and dependency tracking.

    Benefits:
    - Fast cloning without re-initialization
    - Dependency validation before instantiation
    - Version tracking
    - Metadata management

    Example:
        prototype = ModelPrototype(
            model_id="catboost_v1",
            model_class=CatBoostModel,
            version="1.0.0",
            dependencies=["catboost", "numpy"],
            metadata={"iterations": 100, "depth": 6}
        )

        model1 = prototype.clone(iterations=200)
        model2 = prototype.clone(depth=8)
    """

    def __init__(
        self,
        model_id: str,
        model_class: type[BaseModel],
        version: str = "1.0.0",
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize model prototype.

        Args:
            model_id: Unique identifier for the prototype
            model_class: Model class to instantiate
            version: Version string (default: "1.0.0")
            dependencies: List of required package names
            metadata: Default metadata/parameters for model
        """
        self.model_id = model_id
        self.model_class = model_class
        self.version = version
        self.dependencies = dependencies or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self._validated = False
        self._clone_count = 0

    def clone(self, **kwargs) -> BaseModel | None:
        """
        Clone prototype with optional parameter overrides.

        Args:
            **kwargs: Parameters to override from metadata

        Returns:
            New model instance or None if dependencies missing

        Example:
            model = prototype.clone(iterations=500, learning_rate=0.01)
        """
        if not self._validated:
            if not self.validate_dependencies():
                logger.error(f"Cannot clone {self.model_id}: dependencies missing")
                return None
            self._validated = True

        # Merge metadata with overrides
        params = {**self.metadata, **kwargs}

        try:
            model = self.model_class(**params)
            self._clone_count += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Cloned {self.model_id} (#{self._clone_count}) with params: {kwargs}"
                )
            return model
        except Exception as e:
            logger.error(f"Failed to clone {self.model_id}: {e}")
            raise RuntimeError(f"Failed to clone model prototype {self.model_id}") from e

    def validate_dependencies(self) -> bool:
        """
        Validate that all required dependencies are installed.

        Returns:
            True if all dependencies available, False otherwise
        """
        for dep in self.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.warning(f"Dependency missing for {self.model_id}: {dep}")
                return False
        return True

    def get_info(self) -> dict[str, Any]:
        """Get prototype information"""
        return {
            "model_id": self.model_id,
            "model_class": self.model_class.__name__,
            "version": self.version,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "validated": self._validated,
            "clone_count": self._clone_count,
        }

    def __repr__(self) -> str:
        return f"ModelPrototype(id={self.model_id}, version={self.version}, clones={self._clone_count})"

    def __str__(self) -> str:
        return f"{self.model_id} v{self.version}"
