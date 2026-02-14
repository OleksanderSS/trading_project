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
from config.logger_config import LOGGER_CONFIG


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
    _instance: Optional[logging.Logger] = None
    _listener: Optional[QueueListener] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _config: Dict[str, Any] = {}

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Інandцandалandforцandя КОРЕНЕВОГО логера, so that вandн отримував повandдомлення with усandх модулandв."""
        if cls._instance:
            return cls._instance

        # --- ОСТАТОЧНЕ ВИПРАВЛЕННЯ: Налаштовуємо кореnotвий логер ---
        logger = logging.getLogger()  # <-- ОТРИМУЄМО КОРЕНЕВИЙ ЛОГЕР
        logger.setLevel(getattr(logging, LOGGER_CONFIG["level"]))
        logger.handlers.clear()

        log_queue = SimpleQueue()
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                replacements = {
                    '[OK]': '[OK]', '[ERROR]': '[ERROR]', '[WARN]': '[WARN]',
                    '[TARGET]': '[TARGET]', '[UP]': '[CHART]', '[DATA]': '[DATA]',
                    '': '[GLOBAL]', '[BRAIN]': '[BRAIN]', '': '[FILE]',
                    '': '[INFO]'  # Заміна [MSG] на [INFO]
                }
                for emoji, replacement in replacements.items():
                    msg = msg.replace(emoji, replacement)
                return msg

        formatter = SafeFormatter(LOGGER_CONFIG["formatter"])

        import sys
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                pass

        file_handler = None
        if LOGGER_CONFIG["log_file"]:
            log_dir = os.path.dirname(LOGGER_CONFIG["log_file"])
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(LOGGER_CONFIG["log_file"])
            file_handler.setFormatter(formatter)

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
        cls._listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
        cls._listener.start()

        logger.info(f"[OK] Global logger started")

        cls._instance = logger
        cls._config = LOGGER_CONFIG
        return logger

    @classmethod
    def shutdown(cls):
        if cls._instance:
            try:
                cls._instance.critical("[Logger] Shutdown process started...")
            except Exception:
                # Використовуємо logging напряму, якщо andнсandнс логера вже not працює
                logging.warning("[[WARN] Logging error] Неможливо forписати критичний лог при forвершеннand")

        if cls._listener:
            cls._listener.stop()
            cls._listener = None

        if cls._executor:
            cls._executor.shutdown(wait=True)
            cls._executor = None

        cls._instance = None
        cls._config = {}

    # ... (решand класу беwith withмandн)
