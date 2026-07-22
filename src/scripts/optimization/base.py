from abc import ABC, abstractmethod
from typing import Any

from src.core.logging.logger import ProjectLogger


class BaseOptimizer(ABC):
    """
    Абстрактний базовий клас для всіх оптимізаторів системи.
    Забезпечує єдиний інтерфейс для оптимізації портфеля та гіперпараметрів.
    """

    def __init__(self):
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self._best_score: float = 0.0

    @property
    def best_score(self) -> float:
        """Повертає найкращий скор оптимізації."""
        return self._best_score

    @best_score.setter
    def best_score(self, value: float) -> None:
        """Встановлює найкращий скор оптимізації."""
        self._best_score = value

    @property
    @abstractmethod
    def optimizer_type(self) -> str:
        """Повертає тип оптимізатора (наприклад, 'portfolio', 'hyperparameter')."""
        pass

    @abstractmethod
    def optimize(self, data: Any, target: Any = None, **kwargs) -> dict[str, Any]:
        """
        Виконує процес оптимізації.

        Args:
            data: Вхідні дані для оптимізації (DataFrame з цінами, об'єкт моделі тощо).
            target: Цільова змінна (необов'язково, для ML моделей).
            **kwargs: Додаткові параметри оптимізації.

        Returns:
            dict[str, Any]: Результати оптимізації (ваги, параметри, метрики).
        """
        pass

    def validate_params(self, params: dict[str, Any]) -> bool:
        """
        Перевіряє валідність вхідних параметрів.

        Args:
            params: Словник параметрів для перевірки.

        Returns:
            bool: True, якщо параметри валідні.
        """
        if not isinstance(params, dict):
            self.logger.error("Параметри повинні бути словником.")
            return False
        return True

    def get_status(self) -> dict[str, Any]:
        """Повертає поточний статус оптимізатора."""
        return {
            "name": self.__class__.__name__,
            "type": self.optimizer_type
        }
