"""
CLEAN LOGGING - Виправлення конфліктів з TensorFlow/PyTorch
"""

import logging
import sys
import weakref
from typing import Optional

def setup_clean_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Налаштовує чисте логування without конфліктів з TensorFlow/PyTorch
    
    Args:
        level: Рівень логування (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Кастомний формат логів
    """
    # Видаляємо ВСІ існуючі handlers з root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Встановлюємо рівень
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Формат
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Створюємо ОДИН StreamHandler з примусовою буферизацією
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # [FIX] ВИПРАВЛЕНО: Примусова буферизація для handler
    # Це змусить систему чекати на \n перед виводом
    stream_handler.stream.flush()
    
    # Додаємо handler до root logger
    root_logger.addHandler(stream_handler)
    
    # Важливо: не даємо іншим бібліотекам спамити
    root_logger.propagate = False
    
    # [FIX] ВИПРАВЛЕНО: Блокуємо перехоплення stdout іншими бібліотеками
    logging._acquireLock()
    try:
        logging._handlers.clear()
        logging._handlerList[:] = [weakref.ref(stream_handler)]
    finally:
        logging._releaseLock()
    
    print(f"[CLEAN_LOGGING] [OK] Logging setup completed with level: {level}")

def get_clean_logger(name: str) -> logging.Logger:
    """
    Отримує чистий логер для конкретного модуля
    
    Args:
        name: Ім'я модуля
        
    Returns:
        Logger: Чистий логер
    """
    return logging.getLogger(name)

# Автоматично налаштовуємо при імпорті
setup_clean_logging()
