from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.core.logging.logger import ProjectLogger

class BaseOptimizer(ABC):
    """
    Абстрактний базовий клас для всіх оптимізаторів системи.
    Забезпечує єдиний інтерфейс для оптимізації портфеля та гіперпараметрів.
    """

    def __init__(self):
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    @property
    @abstractmethod
    def optimizer_type(self) -> str:
        """Повертає тип оптимізатора (наприклад, 'portfolio', 'hyperparameter')."""
        pass

    @abstractmethod
    def optimize(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Виконує процес оптимізації.

        Args:
            data: Вхідні дані для оптимізації (DataFrame з цінами, об'єкт моделі тощо).
            **kwargs: Додаткові параметри оптимізації.

        Returns:
            Dict[str, Any]: Результати оптимізації (ваги, параметри, метрики).
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
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

    def get_status(self) -> Dict[str, Any]:
        """Повертає поточний статус оптимізатора."""
        return {
            "name": self.__class__.__name__,
            "type": self.optimizer_type
        }