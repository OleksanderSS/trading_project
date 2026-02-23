# config/secrets_manager.py

import os
import threading
import logging
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger("SecretsManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False

# [BRAIN] Loading .env
def load_env_file(path: str = None):
    if path is None:
        current_dir = Path(__file__).parent
        path = current_dir.parent / ".env"

    if os.path.exists(path):
        load_dotenv(dotenv_path=path, override=False)
        logger.info(f"[SecretsManager] [OK] .env loaded from {path}")
    else:
        logger.warning(f"[SecretsManager] [WARN] .env not found at path: {path}")

load_env_file()

class Secrets:
    _instance = None
    _lock = threading.Lock()

    REQUIRED_KEYS = ["FRED_API_KEY", "NEWS_API_KEY"]

    ACTIVE_KEYS = [
        "FRED_API_KEY",
        "NEWS_API_KEY",
        "HF_TOKEN",
        "TRADER_MODE",
        "TRADER_INITIAL_BALANCE",
        "TRADER_RISK_FRACTION",
        "TELEGRAM_TOKEN"
    ]

    FUTURE_KEYS = [
        "TELEGRAM_CHAT_ID",   # planned to add later
        "USE_MEMORY_DB"       # option for future cache
    ]

    ALL_KEYS = ACTIVE_KEYS + FUTURE_KEYS

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
                cls._instance._load()
        return cls._instance

    def _load(self):
        if self._loaded:
            return

        missing_required = []
        for key in self.ALL_KEYS:
            value = os.getenv(key)
            setattr(self, key, value)
            if key in self.REQUIRED_KEYS and not value:
                missing_required.append(key)

        if missing_required:
            raise RuntimeError(f"[SecretsManager] [ERROR] Missing required keys: {missing_required}")

        # Soft notification for FUTURE_KEYS
        inactive = [k for k in self.FUTURE_KEYS if not getattr(self, k)]
        if inactive:
            logger.info(f"[SecretsManager] Keys not yet active (can add later): {inactive}")

        logger.info(f"[SecretsManager] [OK] Loaded {len(self.ALL_KEYS)} keys (active + future)")
        self._loaded = True

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def has(self, key: str) -> bool:
        return getattr(self, key, None) is not None

    def as_dict(self):
        return {k: getattr(self, k) for k in self.ALL_KEYS}
