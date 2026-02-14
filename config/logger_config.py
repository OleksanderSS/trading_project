# config/logger_config.py

LOGGER_CONFIG = {
    "name": "TradingProjectLogger",
    "level": "INFO",
    "log_file": "logs/project.log",
    "telegram_bot_token": None,   # or твandй токен
    "telegram_chat_id": None,     # or твandй chat_id
    "max_threads": 5,
    "formatter": "%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s",
    "telegram_rate_limit": 0.1
}