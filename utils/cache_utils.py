# utils/cache_utils.py

import pandas as pd
import hashlib
import os
import pickle
import time
import logging
from utils.logger import ProjectLogger


logger = ProjectLogger.get_logger("TradingProjectLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


class CacheManager:
    """
    Меnotджер кешу:
    - DataFrame -> Parquet;
    - pickle-об'єкти for довготривалих data;
    - in-memory хешand новин + withбереження на диск;
    - TTL for ключandв.
    """

    DEFAULT_TF_MINUTES = {
        "5m": 5, "15m": 15, "60m": 60, "1d": 1440
    }

    def __init__(self, base_path="cache", tf_ttl=None, default_ttl=1440):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.tf_ttl = tf_ttl or {}
        self.default_ttl = default_ttl
        self._in_memory_hashes = set()
        self.load_news_hashes()

    # ------------------- Хешand -------------------
    def hash_content(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_hash(self, content: str):
        self._in_memory_hashes.add(self.hash_content(content))

    def is_duplicate_news(self, content: str) -> bool:
        return self.hash_content(content) in self._in_memory_hashes

    # ------------------- TTL -------------------
    def _is_expired(self, path: str, ttl_minutes: int) -> bool:
        if not os.path.exists(path):
            return True
        age_minutes = (time.time() - os.path.getmtime(path)) / 60
        return age_minutes > ttl_minutes

    def _get_ttl(self, key: str) -> int:
        for tf, ttl in self.tf_ttl.items():
            if tf in key:
                return ttl
        return self.default_ttl

    # ------------------- UTC cleaner -------------------
    def ensure_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df_out = df.copy()
        if isinstance(df_out.index, pd.DatetimeIndex) and df_out.index.tz is not None:
            df_out.index = df_out.index.tz_convert(None)
        for col in df_out.select_dtypes(include=["datetimetz"]).columns:
            df_out[col] = df_out[col].dt.tz_convert(None)
        return df_out

    # ------------------- DataFrame -------------------
    def set_df(self, key, df: pd.DataFrame):
        df_to_save = self.ensure_utc(df)
        path = os.path.join(self.base_path, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(df_to_save, f)

    def get_df(self, key, fallback=None) -> pd.DataFrame:
        path = os.path.join(self.base_path, f"{key}.pkl")
        ttl = self._get_ttl(key)
        if os.path.exists(path) and not self._is_expired(path, ttl):
            try:
                logger.debug(f"[Cache] Used cache: {key}")
                with open(path, "rb") as f:
                    df = pickle.load(f)
                return self.ensure_utc(df)
            except Exception as e:
                logger.warning(f"[Cache] Error чиandння кешу {key}: {e}")
                return fallback if fallback is not None else pd.DataFrame()
        logger.debug(f"[Cache] Cache expired or missing: {key}")
        return fallback if fallback is not None else pd.DataFrame()

    # ------------------- Pickle -------------------
    def save_pickle(self, key, obj):
        path = os.path.join(self.base_path, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, key, fallback=None):
        path = os.path.join(self.base_path, f"{key}.pkl")
        ttl = self._get_ttl(key)
        if os.path.exists(path) and not self._is_expired(path, ttl):
            try:
                logger.debug(f"[Cache] Used pickle cache: {key}")
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.exception(f"[Cache] Error чиandння pickle {key}: {e}")
                return fallback
        logger.debug(f"[Cache] Pickle cache expired or missing: {key}")
        return fallback

    # ------------------- Alias-методи -------------------
    def load(self, key: str, fallback=None):
        return self.load_pickle(key, fallback)

    def save(self, key: str, obj):
        self.save_pickle(key, obj)

    # ------------------- Системнand методи -------------------
    def save_news_hashes(self):
        path = os.path.join(self.base_path, "news_hashes.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._in_memory_hashes, f)

    def load_news_hashes(self):
        path = os.path.join(self.base_path, "news_hashes.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._in_memory_hashes = pickle.load(f)
            except Exception:
                self._in_memory_hashes = set()

    def clear_cache(self, key_mask=None, clear_memory=False):
        if clear_memory:
            self._in_memory_hashes.clear()
        for f in os.listdir(self.base_path):
            if key_mask is None or key_mask in f:
                os.remove(os.path.join(self.base_path, f))

    def repair_cache(self):
        for f in os.listdir(self.base_path):
            path = os.path.join(self.base_path, f)
            try:
                if f.endswith(".parquet") or f.endswith(".pkl"):
                    with open(path, "rb") as pkl:
                        pickle.load(pkl)
                elif f.endswith(".pkl"):
                    with open(path, "rb") as pkl:
                        pickle.load(pkl)
            except Exception:
                os.remove(path)
