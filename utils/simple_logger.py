# Simple logger without Ukrainian text
import logging
import sys
from typing import Optional, Dict

class SimpleLogger:
    _instances: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Simple logger without conflicts"""
        logger_name = name or "SimpleLogger"
        
        if logger_name in cls._instances:
            return cls._instances[logger_name]
        
        logger = logging.getLogger(logger_name)
        
        # If handlers exist, don't add new ones
        if logger.handlers:
            cls._instances[logger_name] = logger
            return logger
        
        # Simple formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # One simple StreamHandler
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

# Alias for compatibility
ProjectLogger = SimpleLogger
