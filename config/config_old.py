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

# --- Tickers (централandwithованand) ---
# Для сумandсностand with andснуючим codeом
# TICKERS вже andмпортовано with tickers.py (4 основнand тandкери)
# ALL_TICKERS_DICT - повний словник (119 тandкерandв)
# get_tickers() - функцandя for отримання списку for категорandєю
# get_tickers_dict() - функцandя for отримання словника for категорandєю

# --- Timeframes ---
TIME_FRAMES = {
    "15m": {"period": "60d"},
    "60m": {"period": "60d"}, 
    "1d": {"period": "2y"}
}

YF_MAX_PERIODS = {
    "15m": "60d",   # 60 days for 15-minute candles
    "60m": "60d",   # 60 days for hourly candles
    "1d": "2y"      # 2 years for daily candles
}

DATA_INTERVALS = {
    "15m": "15m",
    "60m": "60m",
    "1d": "1d"
}

USE_CORE_FEATURES = True

# --- Дandапаwithони дат ---
def get_date_range(api_name: str):
    today = datetime.now(timezone.utc).replace(microsecond=0)
    api_name = api_name.lower()

    if api_name == "newsapi":
        start = today - timedelta(days=60)  # Змandnotно на 60 днandв for унandфandкацandї
    elif api_name == "fred":
        start = today - timedelta(days=365)
    elif api_name == "financial":
        start = today - timedelta(days=730)
    elif api_name == "news_intraday":   # новий режим for новин пandд 15m/60m
        start = today - timedelta(days=60)
    else:
        start = today - timedelta(days=90)

    return start, today

START_FINANCIAL, END_FINANCIAL = get_date_range("financial")
START_NEWSAPI, END_NEWSAPI = get_date_range("newsapi")
START_NEWS_INTRADAY, END_NEWS_INTRADAY = get_date_range("news_intraday")  #  новини на 2 мandсяцand

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
    logger.info("[Config] [BRAIN] Використовується in-memory баfor data")