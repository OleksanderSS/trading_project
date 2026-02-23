# utils/logger.py

import logging
from logging.handlers import QueueHandler, QueueListener
from queue import SimpleQueue
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os
import requests
from typing import Optional, Dict, Any, List
import sys

# Конфігурація логера за замовчуванням
LOGGER_CONFIG = {
    "name": "ProjectLogger",
    "log_file": "logs/project.log",
    "telegram_bot_token": None,
    "telegram_chat_id": None,
    "telegram_rate_limit": 0.1,
    "max_threads": 2
}


class NonBlockingHTTPHandler(logging.Handler):
    def __init__(self, executor: ThreadPoolExecutor):
        super().__init__()
        self.executor = executor

    def emit(self, record):
        msg = self.format(record)
        self.executor.submit(self._send, msg)

    def _send(self, msg: str):
        raise NotImplementedError("Implement in subclass")


class NonBlockingTelegramHandler(NonBlockingHTTPHandler):
    def __init__(self, bot_token: str, chat_id: str, executor: ThreadPoolExecutor, rate_limit: float = 0.1):
        super().__init__(executor)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.rate_limit = rate_limit
        self._last_sent = 0
        self._lock = threading.Lock()

    def _send(self, msg: str):
        def send_msg():
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                requests.post(url, data={"chat_id": self.chat_id, "text": msg, "parse_mode": "HTML"}, timeout=5)
            except Exception:
                logging.getLogger("Logger").warning("[WARN] Failed to send Telegram log message")

        with self._lock:
            now = time.time()
            if now - self._last_sent < self.rate_limit:
                delay = self.rate_limit - (now - self._last_sent)
                threading.Timer(delay, send_msg).start()
            else:
                send_msg()
            self._last_sent = time.time()


class ProjectLogger:
    _instances: Dict[str, logging.Logger] = {}  #  ВИПРАВЛЕНО: множиннand логери!
    _listener: Optional[QueueListener] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _config: Dict[str, Any] = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Інціалізація простого логера without конфліктів"""
        logger_name = name if name else LOGGER_CONFIG["name"]
        
        # Перевіряємо чи вже існує логер для цього модуля
        if logger_name in cls._instances:
            return cls._instances[logger_name]
        
        # Створюємо простий логер without QueueHandler
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
        
        log_dir = os.path.dirname(LOGGER_CONFIG["log_file"])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)  # [OK] створюємо директорію automatically

        file_handler = logging.FileHandler(LOGGER_CONFIG["log_file"])
        file_handler.setFormatter(formatter)

        #  Telegram
        telegram_handler = None
        if LOGGER_CONFIG["telegram_bot_token"] and LOGGER_CONFIG["telegram_chat_id"]:
            cls._executor = ThreadPoolExecutor(max_workers=LOGGER_CONFIG["max_threads"])
            telegram_handler = NonBlockingTelegramHandler(
                LOGGER_CONFIG["telegram_bot_token"],
                LOGGER_CONFIG["telegram_chat_id"],
                cls._executor,
                rate_limit=LOGGER_CONFIG["telegram_rate_limit"]
            )
            telegram_handler.setFormatter(formatter)

        handlers = [h for h in (stream_handler, file_handler, telegram_handler) if h]
        # ВИПРАВЛЕНО: Створюємо log_queue якщо потрібно
        log_queue = SimpleQueue()
        cls._listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
        cls._listener.start()

        logger.info(f"[OK] Logger started [{logger_name}]")

        # Зберandгаємо логер в словник
        cls._instances[logger_name] = logger
        cls._config = LOGGER_CONFIG
        return logger

    @classmethod
    def shutdown(cls):
        # ВИПРАВЛЕНО: Закриваємо ВСІ логери
        for logger_name, logger in cls._instances.items():
            try:
                logger.critical("[Logger] Shutdown process started...")
            except Exception:
                logger.info("[[WARN] Logging error] Неможливо forписати критичний лог при forвершеннand")
        
        if cls._listener:
            cls._listener.stop()
            cls._listener = None
            
        if cls._executor:
            cls._executor.shutdown(wait=True)
            cls._executor = None
            
        cls._instances.clear()
        cls._config = {}

    # -------------------------------
    # Додатковand логери for фandчей
    # -------------------------------
    @classmethod
    def log_feature_layers(cls, layers: List[str]):
        # ВИПРАВЛЕНО: Логуємо for кожного логера
        for logger_name, logger in cls._instances.items():
            logger.info(f"[BRAIN] Викорисandнand шари фandчей: {', '.join(layers)}")

    @classmethod
    def log_missing_features(cls, missing: List[str]):
        # ВИПРАВЛЕНО: Логуємо for кожного логера
        for logger_name, logger in cls._instances.items():
            logger.warning(f"[WARN] Вandдсутнand фandчand: {', '.join(missing)}")

    @classmethod
    def log_context_summary(cls, context: Dict[str, Any]):
        # ВИПРАВЛЕНО: Логуємо for кожного логера
        for logger_name, logger in cls._instances.items():
            if not context:
                return
            parts = []
            if "market_phase" in context:
                parts.append(f"[UP] Фаfor ринку: {context['market_phase']}")
            if "trend_alignment" in context:
                parts.append(f"[DATA] Уwithгодження with трендом: {context['trend_alignment']}")
            if "macro_bias" in context:
                parts.append(f" Макро-бandас: {context['macro_bias']}")
            if parts:
                logger.info("[BRAIN] Контекст: " + " | ".join(parts))