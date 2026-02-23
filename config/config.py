# config/config.py

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from utils.mention_utils import safe_get
from config.config_manager import detect_environment, resolve_paths
from config.secrets_manager import Secrets


# --- Logger ---
logger = logging.getLogger("config_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False

# --- Secrets ---
secrets = Secrets()
FRED_API_KEY = secrets.get("FRED_API_KEY", "")
HF_TOKEN = secrets.get("HF_TOKEN", "")

# --- Environment & Paths ---
ENV = detect_environment()
PATHS = resolve_paths(ENV)

os.makedirs(PATHS["data"], exist_ok=True)
os.makedirs(PATHS["models"], exist_ok=True)

# --- Tickers (Розширений список) ---
TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "META": "META",
    "BTC-USD": "BTC-USD"
}

# --- Timeframes (Додано 5m) ---
TIME_FRAMES = {
    "5m": {"period": "60d"}, # Максимальний період для yfinance
    "15m": {"period": "60d"},
    "60m": {"period": "730d"}, # 730d для годинних свічок
    "1d": {"period": "5y"}
}

YF_MAX_PERIODS = {
    "5m": "60d",
    "15m": "60d",
    "60m": "730d",
    "1d": "5y"
}

DATA_INTERVALS = {
    "5m": "5m",
    "15m": "15m",
    "60m": "60m",
    "1d": "1d"
}

USE_CORE_FEATURES = True

# --- Діапазони дат ---
def get_date_range(api_name: str):
    today = datetime.now(timezone.utc).replace(microsecond=0)
    api_name = api_name.lower()

    if api_name == "newsapi":
        start = today - timedelta(days=60)
    elif api_name == "fred":
        start = today - timedelta(days=365 * 2) # 2 роки для макро даних
    elif api_name == "financial":
        start = today - timedelta(days=365 * 5) # 5 років для денних даних
    elif api_name == "news_intraday":
        start = today - timedelta(days=60)
    else:
        start = today - timedelta(days=90)

    return start, today

START_FINANCIAL, END_FINANCIAL = get_date_range("financial")
START_NEWSAPI, END_NEWSAPI = get_date_range("newsapi")
START_NEWS_INTRADAY, END_NEWS_INTRADAY = get_date_range("news_intraday")

# --- Trader Config ---
TRADER_MODE = secrets.get("TRADER_MODE", "csv")
TRADER_CONFIG = {
    "initial_balance": float(secrets.get("TRADER_INITIAL_BALANCE", 10000)),
    "risk_fraction": float(secrets.get("TRADER_RISK_FRACTION", 0.25)),
}

NOTIFIER_CONFIG = {
    "telegram_token": secrets.get("TELEGRAM_TOKEN"),
    "chat_id": secrets.get("TELEGRAM_CHAT_ID"),
}

# --- Database ---
USE_MEMORY_DB = secrets.get("USE_MEMORY_DB", "0") == "1"
DB_PATH = ":memory:" if USE_MEMORY_DB else PATHS["db"]
if USE_MEMORY_DB:
    logger.info("[Config] 🧠 Використовується in-memory база даних")
