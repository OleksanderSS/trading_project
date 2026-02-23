#!/usr/bin/env python3
"""
Центральний конфігураційний файл trading системи
Всі параметри в одному місці для гнучкості
"""

# Імпортуємо нову рефакторинуту конфігурацію
from .trading_config_refactored import (
    TradingConfig, DataConfig, RiskConfig, ModelConfig, 
    NewsConfig, LoggingConfig, APIConfig, BacktestConfig,
    get_config, reload_config
)

# Для зворотної сумісності
__all__ = [
    'TradingConfig', 'DataConfig', 'RiskConfig', 'ModelConfig',
    'NewsConfig', 'LoggingConfig', 'APIConfig', 'BacktestConfig',
    'get_config', 'reload_config'
]
