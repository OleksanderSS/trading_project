# utils/logger_simple.py - ПРОСТИЙ ЛОГЕР БЕЗ КОНФЛІКТІВ

import logging
import sys
from typing import Optional, Dict

class ProjectLogger:
    _instances: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Простий логер without конфліктів"""
        logger_name = name or "ProjectLogger"
        
        if logger_name in cls._instances:
            return cls._instances[logger_name]
        
        logger = logging.getLogger(logger_name)
        
        # Якщо вже є handlers, не додаємо нові
        if logger.handlers:
            cls._instances[logger_name] = logger
            return logger
        
        # Простий форматер
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Один простий StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        cls._instances[logger_name] = logger
        return logger

    @classmethod
    def shutdown(cls):
        for logger in cls._instances.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._instances.clear()
