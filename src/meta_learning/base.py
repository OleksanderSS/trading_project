from abc import ABC, abstractmethod
from typing import Any


class BaseMetaComponent(ABC):
    """
    Abstract base class for all meta-learning components.
    Provides a consistent interface for memory, awareness, and evolution engines.
    """

    @abstractmethod
    def update(self, data: Any) -> None:
        """
        Updates the component's internal state with new data or experience.

        Args:
            data: The new information (e.g., trade results, market features)
                  to be processed by the meta-engine.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for the meta-learning component.
        """
        pass

    def get_state(self) -> dict[str, Any]:
        """
        Returns the current internal state of the component.
        Useful for logging, serialization, or state-based decision making.

        Returns:
            A dictionary representing the component's state.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }
