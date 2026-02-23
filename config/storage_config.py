# config/storage_config.py

import os

BASE_PATH = os.getenv('TRADING_PROJECT_PATH', 'c:/trading_project')

# News Storage Configuration
NEWS_STORAGE_CONFIG = {
    "processed_db": os.path.join(BASE_PATH, "data/databases/processed_news.db"),
    "archive_db": os.path.join(BASE_PATH, "data/databases/archive_news.db"),
    "processed_table": "processed_news",
    "archive_table": "archive_news"
}