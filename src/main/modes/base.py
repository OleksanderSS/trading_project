#!/usr/bin/env python3
"""
Base class for all modes using the project's new standards and UnifiedConfigManager.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger


class BaseMode(ABC):
    """
    Абстрактний базовий клас для всіх режимів роботи системи.
    Забезпечує доступ до конфігурації, логування та спільний інтерфейс виконання.
    """

    def __init__(self, config_manager: UnifiedConfigManager | None = None):
        """
        Ініціалізує базовий режим.

        Args:
            config_manager: Примірник UnifiedConfigManager для доступу до налаштувань.
        """
        from src.config.unified_config_manager import get_current_config
        self.config_manager = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Основний метод виконання режиму. Має бути реалізований у нащадках.

        Args:
            **kwargs: Додаткові параметри для запуску конкретного режиму.

        Returns:
            dict[str, Any]: Результати виконання режиму (статус, метрики тощо).
        """
        pass

    def validate_prerequisites(self) -> bool:
        """
        Перевірка передумов (наявність даних, підключень) перед виконанням.
        Повертає True за замовчуванням.
        """
        return True

    def cleanup(self) -> None:
        """
        Очищення ресурсів після завершення роботи режиму.
        Default no-op implementation - can be overridden by subclasses if needed.
        """
        pass

    def get_mode_info(self) -> dict[str, Any]:
        """
        Повертає базову інформацію про поточний режим.
        """
        return {
            'mode_name': self.__class__.__name__,
            'description': self.__doc__.strip() if self.__doc__ else 'No description available',
            'status': 'ready' if self.validate_prerequisites() else 'failed_prerequisites'
        }
