# collectors/base_collector.py

import sqlite3, hashlib, json, threading, logging, os, re
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from abc import ABC, abstractmethod
from utils.news_harmonizer import harmonize_entry
import pandas as pd
from datetime import datetime

logger = logging.getLogger("trading_project.base_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


def _validate_table_name(name: str) -> bool:
    return bool(re.match(r'^[A-Za-z0-9_]+$', name))


def _hash_item(item: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    item_to_hash = {k: item.get(k) for k in keys} if keys else item
    item_str = json.dumps(item_to_hash, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(item_str.encode("utf-8")).hexdigest()


def make_json_serializable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(elem) for elem in obj]
    return obj


class BaseCollector(ABC):
    """
    Абстрактний баwithовий клас for allх withбирачandв data.
    Забеwithпечує логandку роботи with SQLite (withбереження, кешування).
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        use_cache: bool = True,
        strict: bool = True,
        hash_keys: Optional[List[str]] = None,
        logger=logger,
        **kwargs   #  сюди can передавати будь-якand новand параметри
    ):
        if not _validate_table_name(table_name):
            raise ValueError(f"Некоректна наwithва andблицand: {table_name}")

        self.db_path = db_path
        self.table_name = table_name
        self.use_cache = use_cache
        self.strict = strict
        self.hash_keys = hash_keys or ["published_at", "description", "source", "url"]
        self.logger = logger
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()

        #  withбережемо all додатковand параметри у словник
        self.extra_params = kwargs

    def _init_db(self):
        """Інandцandалandwithує with'єднання with БД and створює andблицю, якщо вона not andснує."""
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)

        with self._lock:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._check_schema()
            self.logger.info(f"Database schema verified.")

    def _check_schema(self):
        """Перевandряє or створює схему andблицand. Якщо бракує колонок  додає them."""
        with self._conn:
            cursor = self._conn.cursor()
            # створюємо баwithову andблицю, якщо її notмає
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT UNIQUE,
                    published_at TIMESTAMP,
                    description TEXT,
                    type TEXT,
                    value REAL,
                    sentiment REAL,
                    source TEXT,
                    url TEXT,
                    raw_data JSON
                )
            """)
            self._conn.commit()

            # перевandряємо andснуючand колонки
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            existing_cols = [row[1] for row in cursor.fetchall()]

            # список потрandбних колонок
            required_cols = {
                "title": "TEXT",
                "result": "TEXT"
            }

            # додаємо вandдсутнand
            for col, col_type in required_cols.items():
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {col} {col_type}")
                    self.logger.info(f"[BaseCollector] [OK] Added column '{col}' to {self.table_name}")

            # додаємо унandкальний andнwhereкс for ключових полandв
            cursor.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_{self.table_name}
                ON {self.table_name}(published_at, description, source, url)
            """)
            self._conn.commit()

    def _filter_by_keywords(self, df, text_col="description"):
        """Фandльтрує DataFrame for ключовими словами."""
        if not hasattr(self, "flat_keywords") or not self.flat_keywords:
            return df
        mask = df[text_col].str.contains("|".join(self.flat_keywords), case=False, na=False)
        return df[mask]

    def _filter_similar(self, df, text_col="description", threshold=0.9):
        """Прибирає дублandкати for текстом (простий варandант)."""
        return df.drop_duplicates(subset=[text_col])

    @contextmanager
    def get_db_conn(self):
        """Контекстний меnotджер for forбеwithпечення потокобеwithпечного доступу до БД."""
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                yield conn
            finally:
                conn.close()

    @abstractmethod
    def collect(self) -> Any:
        """Основний метод withбору data. Повиnotн бути реалandwithований у дочandрнandх класах."""
        pass

    def save(self, records: List[Dict[str, Any]], strict: bool = True) -> int:
        """Зберandгає список forписandв у БД with унandкальним хешем."""
        if not records:
            return 0

        with self._conn:
            before = self._conn.total_changes
            cursor = self._conn.cursor()
            batch = []

            for item in records:
                try:
                    # 1. Гармонandforцandя (forвжди передаємо source)
                    source = item.get("source", "Unknown")
                    item = harmonize_entry(item, source=source)
                except Exception as e:
                    self.logger.warning(f"Error гармонandforцandї forпису: {e}. Запис пропущено.")
                    continue

                # 2. Нормалandforцandя ключових полandв перед хешуванням
                desc = str(item.get("description", "")).strip().lower()
                pub = pd.to_datetime(item.get("published_at"), errors="coerce")
                if pd.notna(pub):
                    pub = pub.isoformat(timespec="seconds")
                url = item.get("url") or None
                source = str(item.get("source", "")).strip().lower()

                h = _hash_item({"published_at": pub, "description": desc, "source": source, "url": url})

                # 3. Пandдготовка дати
                published_at = item.get("published_at")
                if published_at:
                    if isinstance(published_at, datetime):
                        published_at = published_at.isoformat()
                    elif isinstance(published_at, pd.Timestamp):
                        published_at = published_at.isoformat()
                    else:
                        published_at = str(published_at)
                else:
                    if strict:
                        self.logger.warning("Запис пропущено: вandдсутня 'published_at'")
                        continue

                # 4. Значення and сентимент (гнучко)
                try:
                    value = float(item.get("value")) if item.get("value") is not None else None
                except (ValueError, TypeError):
                    value = None

                try:
                    sentiment = float(item.get("sentiment")) if item.get("sentiment") is not None else None
                except (ValueError, TypeError):
                    sentiment = None

                # 5. Іншand ключовand поля (гнучко)
                title = item.get("title", None)
                result = item.get("result", None)

                # 6. Серandалandforцandя
                item_serializable = item.copy()
                if isinstance(item_serializable.get("published_at"), pd.Timestamp):
                    item_serializable["published_at"] = item_serializable["published_at"].isoformat()

                batch.append((
                    h, published_at, title, item.get("description"), item.get("type"),
                    value, sentiment, source, item.get("url"), result,
                    json.dumps(make_json_serializable(item_serializable), ensure_ascii=False)
                ))

            cursor.executemany(f"""
                INSERT OR IGNORE INTO {self.table_name}
                (hash, published_at, title, description, type, value, sentiment, source, url, result, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)

        inserted_count = self._conn.total_changes - before
        self._conn.commit()
        return inserted_count