#!/usr/bin/env python3
"""
Base class for all modes
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from config.trading_config import TradingConfig


class BaseMode(ABC):
    """Базовий клас для всіх режимів роботи"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Основний метод виконання режиму"""
        pass
    
    def validate_prerequisites(self) -> bool:
        """Перевірка передумов для виконання режиму"""
        return True
    
    def cleanup(self) -> None:
        """Очищення ресурсів після виконання"""
        pass
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Інформація про режим"""
        return {
            'name': self.__class__.__name__,
            'description': self.__doc__ or 'No description',
            'config': {
                'tickers_count': len(self.config.data.tickers),
                'timeframes': self.config.data.timeframes,
                'max_positions': self.config.risk.max_positions,
                'risk_per_trade': self.config.risk.risk_per_trade
            }
        }
